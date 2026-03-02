"""TerminalBench Native dataset — loads from the terminal-bench pip package.

Agentic benchmark using the native terminal-bench SDK for task loading
and test-based evaluation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evals.core.dataset import DatasetProvider
from evals.core.types import EvalRecord

try:
    from terminal_bench import Task, TaskPaths
    from terminal_bench import TerminalBenchDataset as _TBDataset

    _HAS_TERMINALBENCH = True
except ImportError:
    _HAS_TERMINALBENCH = False


class TerminalBenchNativeDataset(DatasetProvider):
    """TerminalBench using the native terminal-bench pip package."""

    dataset_id = "terminalbench-native"
    dataset_name = "TerminalBench Native"

    def __init__(
        self,
        name: str = "terminal-bench-core",
        version: str = "0.1.1",
        path: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        n_tasks: Optional[int] = None,
    ) -> None:
        self._name = name
        self._version = version
        self._path = Path(path) if path else None
        self._task_ids = task_ids
        self._n_tasks = n_tasks
        self._records: List[EvalRecord] = []

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not _HAS_TERMINALBENCH:
            raise ImportError(
                "The 'terminal-bench' package is required for "
                "TerminalBenchNativeDataset. "
                "Install it with: pip install terminal-bench"
            )

        tb_kwargs: Dict[str, Any] = {
            "name": self._name,
            "version": self._version,
        }
        if self._path is not None:
            tb_kwargs["path"] = str(self._path)
        if self._task_ids is not None:
            tb_kwargs["task_ids"] = self._task_ids
        if self._n_tasks is not None:
            tb_kwargs["n_tasks"] = self._n_tasks

        tb_dataset = _TBDataset(**tb_kwargs)

        task_paths_list: List[Path] = list(tb_dataset.tasks)

        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(task_paths_list)

        if max_samples is not None:
            task_paths_list = task_paths_list[:max_samples]

        self._records = []
        for idx, task_dir in enumerate(task_paths_list):
            record = self._convert_task(task_dir, idx)
            if record is not None:
                self._records.append(record)

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def _convert_task(
        self, task_dir: Path, idx: int,
    ) -> Optional[EvalRecord]:
        task_paths = TaskPaths(task_dir)
        task = Task(task_paths)

        instruction = str(getattr(task, "instruction", "") or "").strip()
        if not instruction:
            return None

        task_id = str(getattr(task, "id", "") or task_dir.name or f"tbn_{idx}")
        category_val = str(getattr(task, "category", "") or "terminal")

        metadata: Dict[str, Any] = {
            "task_id": task_id,
            "task_dir": str(task_dir),
            "category": category_val,
            "name": getattr(task, "name", None),
            "tags": getattr(task, "tags", None),
            "difficulty": getattr(task, "difficulty", None),
            "timeout": getattr(task, "timeout", None),
        }

        return EvalRecord(
            record_id=f"terminalbench-native-{task_id}",
            problem=instruction,
            reference="",
            category="agentic",
            subject=category_val,
            metadata=metadata,
        )


__all__ = ["TerminalBenchNativeDataset"]
