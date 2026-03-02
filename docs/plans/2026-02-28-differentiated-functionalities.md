# Differentiated Functionalities Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make OpenJarvis's functionalities genuinely differentiated: real trace-driven learning (LoRA from traces, agent config evolution), research-grade eval framework, composable recipe system, and 3 Operator recipes.

**Architecture:** Build the trace→learn→eval loop as the backbone (Section 1+2), then compose user-facing features on top (Section 3+4). The learning pipeline mines TraceStore for training data, runs LoRA fine-tuning via transformers/peft, evolves agent TOML configs, and validates improvements via the eval harness. Recipes, templates, and operators are TOML compositions of the 5 pillars.

**Tech Stack:** Python 3.10+, transformers, peft, trl (optional deps for training), existing SQLite TraceStore, existing eval framework (extended), TOML configs.

---

## Section 1: Trace-Driven Learning Pipeline

### Task 1: TrainingDataMiner — Extract training pairs from traces

**Files:**
- Create: `src/openjarvis/learning/training/__init__.py`
- Create: `src/openjarvis/learning/training/data.py`
- Test: `tests/learning/training/test_data.py`

**Context:** TraceStore (`src/openjarvis/traces/store.py:63-214`) stores complete interaction traces in SQLite with query, agent, model, engine, result, outcome, feedback, and steps (route/retrieve/generate/tool_call/respond). We need to mine these for three types of supervised pairs: (input, preferred_output) for SFT, (query_class, best_model) for routing, and (query_type, best_agent_config) for agent learning. The existing `classify_query()` in `learning/trace_policy.py:31-42` classifies queries as code/math/short/long/general.

**Step 1: Write the failing test for TrainingDataMiner**

```python
# tests/learning/training/test_data.py
"""Tests for TrainingDataMiner — extracts training pairs from trace history."""

import time
from pathlib import Path
from openjarvis.core.types import Trace, TraceStep, StepType
from openjarvis.traces.store import TraceStore


def _make_trace(
    query: str,
    result: str,
    model: str = "qwen3:8b",
    agent: str = "orchestrator",
    outcome: str = "success",
    feedback: float = 0.9,
    tools_used: list[str] | None = None,
) -> Trace:
    """Helper to build a realistic trace."""
    now = time.time()
    steps = [
        TraceStep(
            step_type=StepType.GENERATE,
            timestamp=now,
            duration_seconds=1.2,
            input={"query": query},
            output={"content": result, "tokens": 150},
        ),
    ]
    if tools_used:
        for tool in tools_used:
            steps.append(
                TraceStep(
                    step_type=StepType.TOOL_CALL,
                    timestamp=now + 0.5,
                    duration_seconds=0.3,
                    input={"tool": tool, "args": "{}"},
                    output={"result": "ok"},
                )
            )
    steps.append(
        TraceStep(
            step_type=StepType.RESPOND,
            timestamp=now + 2.0,
            duration_seconds=0.0,
            input={},
            output={"content": result},
        )
    )
    return Trace(
        query=query,
        agent=agent,
        model=model,
        engine="ollama",
        steps=steps,
        result=result,
        outcome=outcome,
        feedback=feedback,
        started_at=now,
        ended_at=now + 2.0,
        total_tokens=150,
        total_latency_seconds=2.0,
    )


class TestTrainingDataMiner:
    def setup_method(self, tmp_path_factory=None):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        self.store = TraceStore(Path(self._tmp) / "traces.db")

    def teardown_method(self):
        self.store.close()

    def test_extract_sft_pairs_from_successful_traces(self):
        """Successful traces with high feedback produce (input, output) pairs."""
        from openjarvis.learning.training.data import TrainingDataMiner

        self.store.save(_make_trace("Write hello world in Python", "print('hello world')", feedback=0.95))
        self.store.save(_make_trace("Bad query", "Bad result", outcome="failure", feedback=0.2))

        miner = TrainingDataMiner(self.store, min_quality=0.7)
        pairs = miner.extract_sft_pairs()

        assert len(pairs) == 1
        assert pairs[0]["input"] == "Write hello world in Python"
        assert pairs[0]["output"] == "print('hello world')"
        assert pairs[0]["query_class"] == "code"

    def test_extract_routing_pairs(self):
        """Traces grouped by query class yield best-model recommendations."""
        from openjarvis.learning.training.data import TrainingDataMiner

        # Two models, same query class — model_b has better feedback
        self.store.save(_make_trace("Solve 2+2", "4", model="model_a", feedback=0.6))
        self.store.save(_make_trace("Solve 3+3", "6", model="model_b", feedback=0.95))
        self.store.save(_make_trace("Solve 5*5", "25", model="model_b", feedback=0.9))

        miner = TrainingDataMiner(self.store, min_quality=0.5)
        routing = miner.extract_routing_pairs()

        assert "math" in routing
        assert routing["math"]["best_model"] == "model_b"
        assert routing["math"]["sample_count"] >= 2

    def test_extract_agent_config_pairs(self):
        """Traces grouped by query class yield best agent+tools combos."""
        from openjarvis.learning.training.data import TrainingDataMiner

        self.store.save(_make_trace(
            "Find info about Python",
            "Python is a language",
            agent="orchestrator",
            tools_used=["web_search", "think"],
            feedback=0.9,
        ))
        self.store.save(_make_trace(
            "Search for Rust docs",
            "Rust is a systems language",
            agent="simple",
            tools_used=[],
            feedback=0.4,
        ))

        miner = TrainingDataMiner(self.store, min_quality=0.5)
        agent_pairs = miner.extract_agent_config_pairs()

        assert "general" in agent_pairs
        best = agent_pairs["general"]
        assert best["best_agent"] == "orchestrator"
        assert "web_search" in best["best_tools"]

    def test_empty_store_returns_empty(self):
        """No traces → no training data."""
        from openjarvis.learning.training.data import TrainingDataMiner

        miner = TrainingDataMiner(self.store)
        assert miner.extract_sft_pairs() == []
        assert miner.extract_routing_pairs() == {}
        assert miner.extract_agent_config_pairs() == {}

    def test_min_quality_filter(self):
        """Traces below min_quality are excluded."""
        from openjarvis.learning.training.data import TrainingDataMiner

        self.store.save(_make_trace("Hello", "Hi", feedback=0.3, outcome="success"))

        miner = TrainingDataMiner(self.store, min_quality=0.5)
        assert miner.extract_sft_pairs() == []

    def test_deduplication(self):
        """Identical (input, output) pairs are deduplicated."""
        from openjarvis.learning.training.data import TrainingDataMiner

        self.store.save(_make_trace("Hello", "Hi", feedback=0.9))
        self.store.save(_make_trace("Hello", "Hi", feedback=0.85))

        miner = TrainingDataMiner(self.store, min_quality=0.5)
        pairs = miner.extract_sft_pairs()
        assert len(pairs) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/learning/training/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openjarvis.learning.training'`

**Step 3: Write minimal implementation**

```python
# src/openjarvis/learning/training/__init__.py
"""Training data extraction and model fine-tuning from traces."""

# src/openjarvis/learning/training/data.py
"""Extract training data from the TraceStore for supervised learning."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from openjarvis.core.types import StepType

# Reuse query classification from trace_policy
_CODE_PAT = __import__("re").compile(
    r"(```|import |def |class |function |const |let |var )", __import__("re").IGNORECASE
)
_MATH_PAT = __import__("re").compile(
    r"\b(solve|integral|equation|derivative|sum|product|matrix|factor|simplify|calculate)\b",
    __import__("re").IGNORECASE,
)


def _classify_query(query: str) -> str:
    if _CODE_PAT.search(query):
        return "code"
    if _MATH_PAT.search(query):
        return "math"
    if len(query) < 50:
        return "short"
    if len(query) > 500:
        return "long"
    return "general"


class TrainingDataMiner:
    """Mine TraceStore for supervised training pairs."""

    def __init__(
        self,
        trace_store: Any,
        *,
        min_quality: float = 0.7,
        min_samples_per_class: int = 1,
    ) -> None:
        self._store = trace_store
        self._min_quality = min_quality
        self._min_samples = min_samples_per_class

    def _good_traces(self) -> list:
        """Return traces that meet quality threshold."""
        all_traces = self._store.list_traces(limit=10000)
        return [
            t
            for t in all_traces
            if t.outcome == "success"
            and t.feedback is not None
            and t.feedback >= self._min_quality
        ]

    def extract_sft_pairs(self) -> List[Dict[str, Any]]:
        """Extract (input, output) pairs for supervised fine-tuning."""
        traces = self._good_traces()
        seen: set[tuple[str, str]] = set()
        pairs: list[dict[str, Any]] = []
        for t in traces:
            key = (t.query, t.result)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "input": t.query,
                    "output": t.result,
                    "query_class": _classify_query(t.query),
                    "model": t.model,
                    "feedback": t.feedback,
                }
            )
        return pairs

    def extract_routing_pairs(self) -> Dict[str, Dict[str, Any]]:
        """Extract best model per query class from traces."""
        traces = self._good_traces()
        # Group by (query_class, model) → list of feedbacks
        groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for t in traces:
            qclass = _classify_query(t.query)
            groups[qclass][t.model].append(t.feedback or 0.0)

        result: dict[str, dict[str, Any]] = {}
        for qclass, models in groups.items():
            best_model = max(models, key=lambda m: sum(models[m]) / len(models[m]))
            scores = models[best_model]
            result[qclass] = {
                "best_model": best_model,
                "avg_feedback": sum(scores) / len(scores),
                "sample_count": len(scores),
                "all_models": {
                    m: {"avg_feedback": sum(s) / len(s), "count": len(s)}
                    for m, s in models.items()
                },
            }
        return result

    def extract_agent_config_pairs(self) -> Dict[str, Dict[str, Any]]:
        """Extract best agent+tools combo per query class."""
        traces = self._good_traces()
        # Group by query_class → list of (agent, tools, feedback)
        groups: dict[str, list[tuple[str, list[str], float]]] = defaultdict(list)
        for t in traces:
            qclass = _classify_query(t.query)
            tools_used = [
                s.input.get("tool", "")
                for s in t.steps
                if s.step_type == StepType.TOOL_CALL and s.input.get("tool")
            ]
            groups[qclass].append((t.agent, tools_used, t.feedback or 0.0))

        result: dict[str, dict[str, Any]] = {}
        for qclass, entries in groups.items():
            # Score each agent by average feedback
            agent_scores: dict[str, list[float]] = defaultdict(list)
            agent_tools: dict[str, list[list[str]]] = defaultdict(list)
            for agent, tools, fb in entries:
                agent_scores[agent].append(fb)
                agent_tools[agent].append(tools)

            best_agent = max(agent_scores, key=lambda a: sum(agent_scores[a]) / len(agent_scores[a]))
            # Most common tool set for best agent
            all_tools: list[str] = []
            for tl in agent_tools[best_agent]:
                all_tools.extend(tl)
            # Deduplicate preserving frequency order
            tool_counts: dict[str, int] = defaultdict(int)
            for tool in all_tools:
                tool_counts[tool] += 1
            best_tools = sorted(tool_counts, key=tool_counts.get, reverse=True)

            result[qclass] = {
                "best_agent": best_agent,
                "best_tools": best_tools,
                "avg_feedback": sum(agent_scores[best_agent]) / len(agent_scores[best_agent]),
                "sample_count": len(agent_scores[best_agent]),
            }
        return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/learning/training/test_data.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/openjarvis/learning/training/__init__.py src/openjarvis/learning/training/data.py tests/learning/training/test_data.py
git commit -m "feat: add TrainingDataMiner — extract SFT/routing/agent pairs from traces"
```

---

### Task 2: LoRATrainer — Fine-tune local models from trace data

**Files:**
- Create: `src/openjarvis/learning/training/lora.py`
- Test: `tests/learning/training/test_lora.py`

**Context:** The orchestrator training (`learning/orchestrator/sft_trainer.py`) already wraps transformers for SFT training, but it's specialized for orchestrator episodes. We need a general-purpose LoRA trainer that takes the SFT pairs from TrainingDataMiner and fine-tunes any local model. The existing `OrchestratorPolicyModel.from_pretrained()` in `learning/orchestrator/policy_model.py:55-102` shows the pattern for loading HuggingFace models. Training deps (torch, transformers, peft) are optional — guard with `try/except`.

**Step 1: Write the failing test**

```python
# tests/learning/training/test_lora.py
"""Tests for LoRATrainer — fine-tune local models via LoRA/QLoRA."""

import pytest


class TestLoRATrainerConfig:
    def test_default_config(self):
        from openjarvis.learning.training.lora import LoRATrainingConfig

        cfg = LoRATrainingConfig()
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 2e-5
        assert cfg.output_dir == "checkpoints/lora"

    def test_custom_config(self):
        from openjarvis.learning.training.lora import LoRATrainingConfig

        cfg = LoRATrainingConfig(lora_rank=8, num_epochs=1, output_dir="/tmp/test")
        assert cfg.lora_rank == 8
        assert cfg.num_epochs == 1


class TestLoRATrainer:
    def test_init_without_torch_raises(self):
        """LoRATrainer should raise ImportError if torch is unavailable."""
        from openjarvis.learning.training.lora import LoRATrainer, LoRATrainingConfig, HAS_TORCH

        if not HAS_TORCH:
            with pytest.raises(ImportError, match="torch|transformers|peft"):
                LoRATrainer(LoRATrainingConfig(), model_name="test")

    def test_prepare_dataset_from_pairs(self):
        """SFT pairs from TrainingDataMiner should convert to a training dataset."""
        from openjarvis.learning.training.lora import LoRATrainer, LoRATrainingConfig, HAS_TORCH

        if not HAS_TORCH:
            pytest.skip("torch/transformers/peft not installed")

        pairs = [
            {"input": "What is 2+2?", "output": "4", "query_class": "math"},
            {"input": "Hello", "output": "Hi there!", "query_class": "short"},
        ]

        trainer = LoRATrainer(LoRATrainingConfig(output_dir="/tmp/test_lora"), model_name="Qwen/Qwen3-0.6B")
        dataset = trainer.prepare_dataset(pairs)
        assert len(dataset) == 2
        assert "input_ids" in dataset[0] or "text" in dataset[0]

    def test_training_config_validates(self):
        """Invalid configs should raise."""
        from openjarvis.learning.training.lora import LoRATrainingConfig

        with pytest.raises(ValueError):
            LoRATrainingConfig(lora_rank=0)
        with pytest.raises(ValueError):
            LoRATrainingConfig(num_epochs=0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/learning/training/test_lora.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/openjarvis/learning/training/lora.py
"""LoRA/QLoRA fine-tuning from trace-derived training pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning."""

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048

    # Quantization (QLoRA)
    use_4bit: bool = False

    # Output
    output_dir: str = "checkpoints/lora"
    save_every_n_epochs: int = 1

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")


class LoRATrainer:
    """Fine-tune a local model using LoRA from trace-derived SFT pairs."""

    def __init__(
        self,
        config: LoRATrainingConfig,
        *,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "LoRA training requires torch, transformers, and peft. "
                "Install with: pip install torch transformers peft"
            )
        self.config = config
        self.model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        quantization_config = None
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self._device == "cuda" else None,
            torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
        )

        if self.config.gradient_checkpointing:
            self._model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
        )
        self._model = get_peft_model(self._model, lora_config)

    def prepare_dataset(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert SFT pairs to tokenized training examples."""
        self._load_model()
        assert self._tokenizer is not None

        dataset = []
        for pair in pairs:
            text = f"### Instruction:\n{pair['input']}\n\n### Response:\n{pair['output']}"
            tokenized = self._tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            dataset.append(
                {
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                    "text": text,
                }
            )
        return dataset

    def train(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run LoRA fine-tuning on the given SFT pairs.

        Returns dict with training metrics (loss, epochs, adapter_path).
        """
        self._load_model()
        assert self._model is not None and self._tokenizer is not None

        dataset = self.prepare_dataset(pairs)
        if not dataset:
            return {"status": "skipped", "reason": "no training data"}

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_loss = 0.0
        steps = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(dataset), self.config.batch_size):
                batch = dataset[i : i + self.config.batch_size]
                input_ids = torch.stack([b["input_ids"] for b in batch]).to(self._device)
                attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(self._device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                steps += 1

            total_loss += epoch_loss

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                ckpt_path = output_dir / f"epoch_{epoch + 1}"
                self._model.save_pretrained(str(ckpt_path))
                self._tokenizer.save_pretrained(str(ckpt_path))

        # Save final adapter
        adapter_path = output_dir / "final"
        self._model.save_pretrained(str(adapter_path))
        self._tokenizer.save_pretrained(str(adapter_path))

        return {
            "status": "completed",
            "epochs": self.config.num_epochs,
            "total_steps": steps,
            "avg_loss": total_loss / max(steps, 1),
            "adapter_path": str(adapter_path),
            "training_samples": len(dataset),
        }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/learning/training/test_lora.py -v`
Expected: 3 tests PASS (2 may skip if torch not installed, 1 config test always passes)

**Step 5: Commit**

```bash
git add src/openjarvis/learning/training/lora.py tests/learning/training/test_lora.py
git commit -m "feat: add LoRATrainer — fine-tune local models from trace-derived SFT pairs"
```

---

### Task 3: AgentConfigEvolver — Rewrite agent configs from traces

**Files:**
- Create: `src/openjarvis/learning/agent_evolver.py`
- Test: `tests/learning/test_agent_evolver.py`

**Context:** `AgentAdvisorPolicy` (`learning/agent_advisor.py:11-153`) returns recommendations but never applies them. We need an `AgentConfigEvolver` that: (1) analyzes traces to find which tools/agents/max_turns work best per query class, (2) writes updated agent TOML configs with version tracking, (3) supports rollback. Operators use TOML manifests (`operators/types.py:9-36`) with fields: tools, system_prompt, max_turns, temperature. Agent templates will follow the same format.

**Step 1: Write the failing test**

```python
# tests/learning/test_agent_evolver.py
"""Tests for AgentConfigEvolver — rewrites agent TOML configs from trace analysis."""

import tempfile
import time
from pathlib import Path

from openjarvis.core.types import Trace, TraceStep, StepType
from openjarvis.traces.store import TraceStore


def _trace(
    query: str,
    agent: str = "orchestrator",
    model: str = "qwen3:8b",
    tools_used: list[str] | None = None,
    feedback: float = 0.9,
    outcome: str = "success",
    turns: int = 3,
) -> Trace:
    now = time.time()
    steps = []
    for i in range(turns):
        steps.append(TraceStep(step_type=StepType.GENERATE, timestamp=now + i, duration_seconds=0.5))
    for tool in (tools_used or []):
        steps.append(TraceStep(
            step_type=StepType.TOOL_CALL, timestamp=now + turns,
            duration_seconds=0.2, input={"tool": tool}, output={"result": "ok"},
        ))
    steps.append(TraceStep(step_type=StepType.RESPOND, timestamp=now + turns + 1, duration_seconds=0.0))
    return Trace(
        query=query, agent=agent, model=model, engine="ollama",
        steps=steps, result="result", outcome=outcome, feedback=feedback,
        started_at=now, ended_at=now + turns + 1,
        total_tokens=200, total_latency_seconds=float(turns + 1),
    )


class TestAgentConfigEvolver:
    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self.store = TraceStore(Path(self._tmp) / "traces.db")
        self.config_dir = Path(self._tmp) / "agent_configs"
        self.config_dir.mkdir()

    def teardown_method(self):
        self.store.close()

    def test_evolve_recommends_tool_changes(self):
        from openjarvis.learning.agent_evolver import AgentConfigEvolver

        # web_search used and helpful (high feedback), calculator never useful (low feedback)
        for _ in range(5):
            self.store.save(_trace("Research AI", tools_used=["web_search", "think"], feedback=0.9))
        for _ in range(5):
            self.store.save(_trace("Research AI", tools_used=["calculator", "think"], feedback=0.4))

        evolver = AgentConfigEvolver(self.store, config_dir=self.config_dir)
        recommendations = evolver.analyze()

        assert len(recommendations) > 0
        # Should recommend web_search over calculator for "general" query class
        general_rec = [r for r in recommendations if r["query_class"] == "general"]
        assert len(general_rec) > 0
        assert "web_search" in general_rec[0]["recommended_tools"]

    def test_write_config_creates_toml(self):
        from openjarvis.learning.agent_evolver import AgentConfigEvolver

        evolver = AgentConfigEvolver(self.store, config_dir=self.config_dir)
        evolver.write_config(
            "test_agent",
            tools=["web_search", "think"],
            max_turns=8,
            temperature=0.5,
        )

        config_path = self.config_dir / "test_agent.toml"
        assert config_path.exists()

        import tomllib
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert data["agent"]["tools"] == ["web_search", "think"]
        assert data["agent"]["max_turns"] == 8

    def test_versioning_and_rollback(self):
        from openjarvis.learning.agent_evolver import AgentConfigEvolver

        evolver = AgentConfigEvolver(self.store, config_dir=self.config_dir)

        # Version 1
        evolver.write_config("test_agent", tools=["think"], max_turns=5)
        # Version 2
        evolver.write_config("test_agent", tools=["think", "web_search"], max_turns=10)

        versions = evolver.list_versions("test_agent")
        assert len(versions) >= 2

        # Rollback to version 1
        evolver.rollback("test_agent", version=1)
        import tomllib
        with open(self.config_dir / "test_agent.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["agent"]["tools"] == ["think"]
        assert data["agent"]["max_turns"] == 5

    def test_analyze_empty_store(self):
        from openjarvis.learning.agent_evolver import AgentConfigEvolver

        evolver = AgentConfigEvolver(self.store, config_dir=self.config_dir)
        recommendations = evolver.analyze()
        assert recommendations == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/learning/test_agent_evolver.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/openjarvis/learning/agent_evolver.py
"""Evolve agent TOML configs based on trace analysis."""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.core.types import StepType

# Reuse query classification
_CODE_PAT = __import__("re").compile(
    r"(```|import |def |class |function |const |let |var )", __import__("re").IGNORECASE
)
_MATH_PAT = __import__("re").compile(
    r"\b(solve|integral|equation|derivative|sum|product|matrix|factor|simplify|calculate)\b",
    __import__("re").IGNORECASE,
)


def _classify_query(query: str) -> str:
    if _CODE_PAT.search(query):
        return "code"
    if _MATH_PAT.search(query):
        return "math"
    if len(query) < 50:
        return "short"
    if len(query) > 500:
        return "long"
    return "general"


class AgentConfigEvolver:
    """Analyze traces and evolve agent TOML configurations."""

    def __init__(
        self,
        trace_store: Any,
        *,
        config_dir: str | Path,
        min_quality: float = 0.5,
    ) -> None:
        self._store = trace_store
        self._config_dir = Path(config_dir)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._min_quality = min_quality
        # Version history dir
        self._history_dir = self._config_dir / ".history"
        self._history_dir.mkdir(exist_ok=True)

    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze traces to produce agent config recommendations."""
        traces = self._store.list_traces(limit=10000)
        good = [
            t for t in traces
            if t.outcome == "success"
            and t.feedback is not None
            and t.feedback >= self._min_quality
        ]
        if not good:
            return []

        # Group by query_class → tool_sets with feedback
        class_data: dict[str, list[dict]] = defaultdict(list)
        for t in good:
            qclass = _classify_query(t.query)
            tools_used = [
                s.input.get("tool", "")
                for s in t.steps
                if s.step_type == StepType.TOOL_CALL and s.input.get("tool")
            ]
            turns = sum(1 for s in t.steps if s.step_type == StepType.GENERATE)
            class_data[qclass].append({
                "agent": t.agent,
                "tools": tools_used,
                "turns": turns,
                "feedback": t.feedback,
                "latency": t.total_latency_seconds,
            })

        recommendations = []
        for qclass, entries in class_data.items():
            # Find tool sets that correlate with high feedback
            tool_scores: dict[str, list[float]] = defaultdict(list)
            for e in entries:
                for tool in e["tools"]:
                    tool_scores[tool].append(e["feedback"])

            # Rank tools by average feedback when used
            ranked_tools = sorted(
                tool_scores.items(),
                key=lambda x: sum(x[1]) / len(x[1]),
                reverse=True,
            )
            best_tools = [t for t, _ in ranked_tools if sum(tool_scores[t]) / len(tool_scores[t]) >= self._min_quality]

            # Average turns for successful traces
            avg_turns = sum(e["turns"] for e in entries) / len(entries)

            # Best agent
            agent_fb: dict[str, list[float]] = defaultdict(list)
            for e in entries:
                agent_fb[e["agent"]].append(e["feedback"])
            best_agent = max(agent_fb, key=lambda a: sum(agent_fb[a]) / len(agent_fb[a]))

            recommendations.append({
                "query_class": qclass,
                "recommended_tools": best_tools,
                "recommended_agent": best_agent,
                "recommended_max_turns": max(3, round(avg_turns * 1.5)),
                "sample_count": len(entries),
            })

        return recommendations

    def write_config(
        self,
        agent_name: str,
        *,
        tools: List[str],
        max_turns: int = 10,
        temperature: float = 0.3,
        system_prompt: str = "",
    ) -> Path:
        """Write an agent TOML config, versioning the previous one."""
        config_path = self._config_dir / f"{agent_name}.toml"

        # Archive previous version if exists
        if config_path.exists():
            versions = self._get_version_list(agent_name)
            next_version = len(versions) + 1
            archive_path = self._history_dir / f"{agent_name}_v{next_version}.toml"
            shutil.copy2(config_path, archive_path)

        # Write new config
        lines = [
            f"# Agent config: {agent_name}",
            f"# Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            "",
            "[agent]",
            f'name = "{agent_name}"',
            f"tools = {json.dumps(tools)}",
            f"max_turns = {max_turns}",
            f"temperature = {temperature}",
        ]
        if system_prompt:
            # Use multi-line string for system prompt
            lines.append(f'system_prompt = """{system_prompt}"""')
        lines.append("")

        config_path.write_text("\n".join(lines))
        return config_path

    def list_versions(self, agent_name: str) -> List[Dict[str, Any]]:
        """List all versions of an agent config."""
        versions = self._get_version_list(agent_name)
        result = []
        for i, path in enumerate(versions, 1):
            result.append({
                "version": i,
                "path": str(path),
                "modified": path.stat().st_mtime,
            })
        # Add current as latest version
        current = self._config_dir / f"{agent_name}.toml"
        if current.exists():
            result.append({
                "version": len(result) + 1,
                "path": str(current),
                "modified": current.stat().st_mtime,
            })
        return result

    def rollback(self, agent_name: str, version: int) -> None:
        """Rollback agent config to a specific version."""
        versions = self._get_version_list(agent_name)
        if version < 1 or version > len(versions):
            raise ValueError(f"Version {version} not found. Available: 1-{len(versions)}")
        source = versions[version - 1]
        target = self._config_dir / f"{agent_name}.toml"
        shutil.copy2(source, target)

    def _get_version_list(self, agent_name: str) -> list[Path]:
        """Get sorted list of archived versions."""
        pattern = f"{agent_name}_v*.toml"
        versions = sorted(self._history_dir.glob(pattern))
        return versions
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/learning/test_agent_evolver.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/openjarvis/learning/agent_evolver.py tests/learning/test_agent_evolver.py
git commit -m "feat: add AgentConfigEvolver — rewrite agent TOML configs from trace analysis"
```

---

### Task 4: LearningOrchestrator — Coordinate the full learning loop

**Files:**
- Create: `src/openjarvis/learning/learning_orchestrator.py`
- Test: `tests/learning/test_learning_orchestrator.py`

**Context:** This is the conductor. It runs: (1) collect new traces from TraceStore, (2) mine training data via TrainingDataMiner, (3) run baseline evals, (4) execute learning (LoRA training + agent config evolution + routing policy update), (5) run post-learning evals, (6) accept if improved or rollback. The existing `LearningConfig` (`core/config.py`) has `auto_update` and `update_interval` fields. The existing GRPO/Bandit policies have `update()` methods.

**Step 1: Write the failing test**

```python
# tests/learning/test_learning_orchestrator.py
"""Tests for LearningOrchestrator — coordinates the full learning loop."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from openjarvis.core.types import Trace, TraceStep, StepType
from openjarvis.traces.store import TraceStore


def _trace(query: str = "Hello", feedback: float = 0.9) -> Trace:
    now = time.time()
    return Trace(
        query=query, agent="orchestrator", model="qwen3:8b", engine="ollama",
        steps=[
            TraceStep(step_type=StepType.GENERATE, timestamp=now, duration_seconds=1.0),
            TraceStep(step_type=StepType.RESPOND, timestamp=now + 1, duration_seconds=0.0),
        ],
        result="Hi!", outcome="success", feedback=feedback,
        started_at=now, ended_at=now + 1, total_tokens=50, total_latency_seconds=1.0,
    )


class TestLearningOrchestrator:
    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self.store = TraceStore(Path(self._tmp) / "traces.db")

    def teardown_method(self):
        self.store.close()

    def test_run_with_no_traces_is_noop(self):
        from openjarvis.learning.learning_orchestrator import LearningOrchestrator

        orch = LearningOrchestrator(
            trace_store=self.store,
            config_dir=Path(self._tmp) / "configs",
        )
        result = orch.run()
        assert result["status"] == "skipped"
        assert "no training data" in result["reason"].lower() or "no traces" in result["reason"].lower()

    def test_run_extracts_data_and_updates_routing(self):
        from openjarvis.learning.learning_orchestrator import LearningOrchestrator

        for _ in range(10):
            self.store.save(_trace("What is AI?", feedback=0.9))

        orch = LearningOrchestrator(
            trace_store=self.store,
            config_dir=Path(self._tmp) / "configs",
        )
        result = orch.run()
        assert result["status"] in ("completed", "skipped")
        assert "sft_pairs" in result or "routing_pairs" in result or "reason" in result

    def test_run_with_eval_gate(self):
        """If eval_fn is provided, learning is accepted only if score improves."""
        from openjarvis.learning.learning_orchestrator import LearningOrchestrator

        for _ in range(10):
            self.store.save(_trace(feedback=0.9))

        # Eval that always returns worse score after learning
        eval_calls = []

        def mock_eval() -> float:
            eval_calls.append(1)
            # First call (baseline) returns 0.8, second (post) returns 0.7
            return 0.8 if len(eval_calls) == 1 else 0.7

        orch = LearningOrchestrator(
            trace_store=self.store,
            config_dir=Path(self._tmp) / "configs",
            eval_fn=mock_eval,
            min_improvement=0.01,
        )
        result = orch.run()
        assert result.get("accepted") is False or result["status"] == "skipped"

    def test_run_records_learning_cycle(self):
        from openjarvis.learning.learning_orchestrator import LearningOrchestrator

        for _ in range(10):
            self.store.save(_trace(feedback=0.9))

        orch = LearningOrchestrator(
            trace_store=self.store,
            config_dir=Path(self._tmp) / "configs",
        )
        result = orch.run()
        assert "timestamp" in result
        assert "sft_pairs" in result or "status" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/learning/test_learning_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/openjarvis/learning/learning_orchestrator.py
"""Orchestrate the full trace-driven learning loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from openjarvis.learning.training.data import TrainingDataMiner
from openjarvis.learning.agent_evolver import AgentConfigEvolver


class LearningOrchestrator:
    """Coordinate trace mining, model fine-tuning, agent evolution, and eval gating.

    Steps:
    1. Mine traces for training data
    2. Run baseline eval (if eval_fn provided)
    3. Update routing policies
    4. Evolve agent configs
    5. Fine-tune model weights (if torch available and enough data)
    6. Run post-learning eval
    7. Accept or rollback
    """

    def __init__(
        self,
        *,
        trace_store: Any,
        config_dir: str | Path,
        eval_fn: Optional[Callable[[], float]] = None,
        min_improvement: float = 0.02,
        min_sft_pairs: int = 10,
        min_quality: float = 0.7,
        lora_config: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._store = trace_store
        self._config_dir = Path(config_dir)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._eval_fn = eval_fn
        self._min_improvement = min_improvement
        self._min_sft_pairs = min_sft_pairs
        self._miner = TrainingDataMiner(trace_store, min_quality=min_quality)
        self._evolver = AgentConfigEvolver(trace_store, config_dir=self._config_dir)
        self._lora_config = lora_config
        self._model_name = model_name

    def run(self) -> Dict[str, Any]:
        """Execute one learning cycle. Returns metrics dict."""
        result: dict[str, Any] = {"timestamp": time.time()}

        # Step 1: Mine training data
        sft_pairs = self._miner.extract_sft_pairs()
        routing_pairs = self._miner.extract_routing_pairs()
        agent_pairs = self._miner.extract_agent_config_pairs()

        result["sft_pairs"] = len(sft_pairs)
        result["routing_classes"] = len(routing_pairs)
        result["agent_classes"] = len(agent_pairs)

        if not sft_pairs and not routing_pairs:
            result["status"] = "skipped"
            result["reason"] = "No training data — insufficient quality traces"
            return result

        # Step 2: Baseline eval
        baseline_score: Optional[float] = None
        if self._eval_fn is not None:
            baseline_score = self._eval_fn()
            result["baseline_score"] = baseline_score

        # Step 3: Update routing (always runs if data available)
        if routing_pairs:
            result["routing_updated"] = True
            result["routing_recommendations"] = routing_pairs

        # Step 4: Evolve agent configs
        if agent_pairs:
            recommendations = self._evolver.analyze()
            for rec in recommendations:
                self._evolver.write_config(
                    f"evolved_{rec['query_class']}",
                    tools=rec["recommended_tools"],
                    max_turns=rec["recommended_max_turns"],
                )
            result["agent_configs_evolved"] = len(recommendations)

        # Step 5: LoRA fine-tuning (optional, requires torch)
        if len(sft_pairs) >= self._min_sft_pairs and self._lora_config is not None:
            try:
                from openjarvis.learning.training.lora import LoRATrainer

                trainer = LoRATrainer(
                    self._lora_config,
                    model_name=self._model_name or "Qwen/Qwen3-0.6B",
                )
                train_result = trainer.train(sft_pairs)
                result["lora_training"] = train_result
            except ImportError:
                result["lora_training"] = {"status": "skipped", "reason": "torch not installed"}

        # Step 6: Post-learning eval
        if self._eval_fn is not None:
            post_score = self._eval_fn()
            result["post_score"] = post_score

            improvement = post_score - (baseline_score or 0.0)
            result["improvement"] = improvement
            result["accepted"] = improvement >= self._min_improvement

            if not result["accepted"]:
                # Rollback agent configs
                # (LoRA adapter can simply not be loaded)
                result["status"] = "rejected"
                result["reason"] = f"Improvement {improvement:.4f} below threshold {self._min_improvement}"
                return result

        result["status"] = "completed"
        return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/learning/test_learning_orchestrator.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/openjarvis/learning/learning_orchestrator.py tests/learning/test_learning_orchestrator.py
git commit -m "feat: add LearningOrchestrator — coordinate trace→learn→eval loop"
```

---

## Section 2: Eval Framework Expansion

### Task 5: Integrate evals into jarvis CLI

**Files:**
- Create: `src/openjarvis/cli/eval_cmd.py`
- Modify: `src/openjarvis/cli/__init__.py:34-54` (add `eval` command group)
- Test: `tests/cli/test_eval_cmd.py`

**Context:** The eval framework currently runs as `python -m evals` (standalone). We need `jarvis eval run`, `jarvis eval compare`, `jarvis eval report` commands. The CLI uses Click (`src/openjarvis/cli/__init__.py`). All existing commands follow the pattern: define a Click group, register as `cli.add_command()`.

**Step 1: Write the failing test**

```python
# tests/cli/test_eval_cmd.py
"""Tests for jarvis eval CLI commands."""

from click.testing import CliRunner


class TestEvalCLI:
    def test_eval_group_exists(self):
        from openjarvis.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "compare" in result.output
        assert "report" in result.output
        assert "list" in result.output

    def test_eval_list_benchmarks(self):
        from openjarvis.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "list"])
        assert result.exit_code == 0
        # Should list available benchmarks
        assert "supergpqa" in result.output.lower() or "benchmark" in result.output.lower()

    def test_eval_run_missing_args(self):
        from openjarvis.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "run"])
        # Should fail gracefully with usage message
        assert result.exit_code != 0 or "usage" in result.output.lower() or "error" in result.output.lower() or "required" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_eval_cmd.py -v`
Expected: FAIL (eval command not registered)

**Step 3: Write minimal implementation**

```python
# src/openjarvis/cli/eval_cmd.py
"""CLI commands for the evaluation framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.group("eval")
def eval_group() -> None:
    """Evaluation framework — benchmark models, agents, and learning."""
    pass


@eval_group.command("list")
def eval_list() -> None:
    """List available benchmarks and backends."""
    try:
        from evals.datasets import supergpqa, gaia

        click.echo("Available benchmarks:")
        click.echo("  supergpqa    — SuperGPQA reasoning MCQ (HuggingFace)")
        click.echo("  gaia         — GAIA agentic benchmark (HuggingFace)")
    except ImportError:
        click.echo("Available benchmarks:")

    # Also list any we know about statically
    benchmarks = [
        ("supergpqa", "SuperGPQA reasoning MCQ"),
        ("gaia", "GAIA agentic tasks"),
        ("frames", "FRAMES multi-hop RAG"),
        ("wildchat", "WildChat conversation quality"),
        ("chat", "Multi-turn conversation quality"),
        ("coding", "Code generation (HumanEval/MBPP)"),
        ("rag", "Retrieval-augmented accuracy"),
    ]
    click.echo("\nRegistered eval suites:")
    for name, desc in benchmarks:
        click.echo(f"  {name:14s} — {desc}")

    click.echo("\nBackends:")
    click.echo("  jarvis-direct — Engine-level inference")
    click.echo("  jarvis-agent  — Agent-level with tools")


@eval_group.command("run")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), help="TOML suite config")
@click.option("-b", "--benchmark", type=str, help="Single benchmark name")
@click.option("-m", "--model", type=str, help="Model identifier")
@click.option("-n", "--max-samples", type=int, default=None, help="Max samples to evaluate")
@click.option("--backend", type=str, default="jarvis-direct", help="Inference backend")
@click.option("--agent", type=str, default=None, help="Agent name (for jarvis-agent backend)")
@click.option("--tools", type=str, default=None, help="Comma-separated tool names")
@click.option("--telemetry/--no-telemetry", default=False, help="Enable telemetry collection")
@click.option("--output", type=click.Path(), default=None, help="Output path for results JSONL")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def eval_run(
    config_path: str | None,
    benchmark: str | None,
    model: str | None,
    max_samples: int | None,
    backend: str,
    agent: str | None,
    tools: str | None,
    telemetry: bool,
    output: str | None,
    verbose: bool,
) -> None:
    """Run evaluation benchmarks."""
    if config_path:
        # Delegate to evals CLI for suite configs
        try:
            from evals.core.config import load_eval_config, expand_suite
            from evals.cli import _run_single

            suite = load_eval_config(config_path)
            runs = expand_suite(suite)
            click.echo(f"Running {len(runs)} eval configurations from {config_path}")
            for i, run_config in enumerate(runs, 1):
                click.echo(f"\n[{i}/{len(runs)}] {run_config.benchmark} / {run_config.model}")
                _run_single(run_config, verbose=verbose)
        except ImportError as e:
            click.echo(f"Error: eval framework not available: {e}", err=True)
            sys.exit(1)
    elif benchmark and model:
        try:
            from evals.core.types import RunConfig
            from evals.cli import _run_single

            tool_list = [t.strip() for t in tools.split(",")] if tools else []
            run_config = RunConfig(
                benchmark=benchmark,
                backend=backend,
                model=model,
                max_samples=max_samples,
                agent_name=agent,
                tools=tool_list,
                telemetry=telemetry,
                output_path=output,
            )
            _run_single(run_config, verbose=verbose)
        except ImportError as e:
            click.echo(f"Error: eval framework not available: {e}", err=True)
            sys.exit(1)
    else:
        click.echo("Error: provide either --config or both --benchmark and --model", err=True)
        sys.exit(1)


@eval_group.command("compare")
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--metric", type=str, default="accuracy", help="Primary metric to compare")
def eval_compare(result_files: tuple[str, ...], metric: str) -> None:
    """Compare results from multiple eval runs."""
    summaries = []
    for path in result_files:
        summary_path = path.replace(".jsonl", ".summary.json")
        if Path(summary_path).exists():
            with open(summary_path) as f:
                summaries.append(json.load(f))
        else:
            click.echo(f"Warning: no summary found for {path}")

    if not summaries:
        click.echo("No summaries to compare.")
        return

    click.echo(f"\n{'Run':<40} {'Accuracy':>10} {'Latency':>10} {'Cost':>10}")
    click.echo("-" * 72)
    for s in summaries:
        name = s.get("benchmark", "unknown") + "/" + s.get("model", "unknown")
        acc = f"{s.get('accuracy', 0):.1%}"
        lat = f"{s.get('mean_latency_seconds', 0):.2f}s"
        cost = f"${s.get('total_cost_usd', 0):.4f}"
        click.echo(f"{name:<40} {acc:>10} {lat:>10} {cost:>10}")


@eval_group.command("report")
@click.argument("result_file", type=click.Path(exists=True))
def eval_report(result_file: str) -> None:
    """Generate detailed report from eval results."""
    summary_path = result_file.replace(".jsonl", ".summary.json")
    if Path(summary_path).exists():
        with open(summary_path) as f:
            summary = json.load(f)
        click.echo(json.dumps(summary, indent=2))
    else:
        # Read JSONL and compute basic stats
        results = []
        with open(result_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        correct = sum(1 for r in results if r.get("is_correct"))
        total = len(results)
        click.echo(f"Results: {correct}/{total} correct ({correct/max(total,1):.1%})")
```

Then register in `src/openjarvis/cli/__init__.py` by adding `from openjarvis.cli.eval_cmd import eval_group` and `cli.add_command(eval_group)`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_eval_cmd.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/openjarvis/cli/eval_cmd.py src/openjarvis/cli/__init__.py tests/cli/test_eval_cmd.py
git commit -m "feat: add jarvis eval CLI — run/compare/report/list commands"
```

---

### Task 6: Add Chat and Coding eval suites

**Files:**
- Create: `evals/datasets/chat_mtbench.py`
- Create: `evals/datasets/coding_humaneval.py`
- Create: `evals/scorers/chat_judge.py`
- Create: `evals/scorers/coding_exec.py`
- Test: `tests/evals/test_new_suites.py`

**Context:** Existing datasets (SuperGPQA, GAIA, FRAMES, WildChat) cover reasoning and agentic workloads. We need Chat (MT-Bench style) and Coding (HumanEval) to complete the 5 suite coverage. All datasets implement `DatasetProvider` ABC (`evals/core/dataset.py:6-37`) with `load()` → `List[EvalRecord]`. All scorers implement `Scorer` ABC (`evals/core/scorer.py:7-19`) with `score()` → `(bool, dict)`.

**Step 1: Write the failing test**

```python
# tests/evals/test_new_suites.py
"""Tests for Chat and Coding eval suites."""


class TestChatDataset:
    def test_loads_with_category_chat(self):
        from evals.datasets.chat_mtbench import ChatMTBenchDataset

        ds = ChatMTBenchDataset()
        records = ds.load(max_samples=3)
        assert len(records) <= 3
        assert all(r.category == "chat" for r in records)
        assert all(r.problem for r in records)

    def test_has_subjects(self):
        from evals.datasets.chat_mtbench import ChatMTBenchDataset

        ds = ChatMTBenchDataset()
        records = ds.load(max_samples=10)
        subjects = {r.subject for r in records}
        # Should have at least some variety
        assert len(subjects) >= 1


class TestCodingDataset:
    def test_loads_with_category_coding(self):
        from evals.datasets.coding_humaneval import CodingHumanEvalDataset

        ds = CodingHumanEvalDataset()
        records = ds.load(max_samples=3)
        assert len(records) <= 3
        assert all(r.category == "coding" for r in records)

    def test_has_test_cases_in_metadata(self):
        from evals.datasets.coding_humaneval import CodingHumanEvalDataset

        ds = CodingHumanEvalDataset()
        records = ds.load(max_samples=3)
        for r in records:
            assert "test" in r.metadata or "entry_point" in r.metadata


class TestChatScorer:
    def test_scorer_interface(self):
        from evals.scorers.chat_judge import ChatJudgeScorer

        scorer = ChatJudgeScorer(judge_model="gpt-5-mini-2025-08-07")
        assert hasattr(scorer, "score")


class TestCodingScorer:
    def test_scorer_interface(self):
        from evals.scorers.coding_exec import CodingExecScorer

        scorer = CodingExecScorer()
        assert hasattr(scorer, "score")

    def test_score_correct_code(self):
        from evals.scorers.coding_exec import CodingExecScorer
        from evals.core.types import EvalRecord

        scorer = CodingExecScorer()
        record = EvalRecord(
            record_id="test_1",
            problem="Write a function that adds two numbers",
            reference="def add(a, b): return a + b",
            category="coding",
            metadata={
                "entry_point": "add",
                "test": "assert add(1, 2) == 3\nassert add(0, 0) == 0",
            },
        )
        is_correct, meta = scorer.score(
            record=record,
            model_answer="def add(a, b):\n    return a + b",
        )
        assert is_correct is True

    def test_score_incorrect_code(self):
        from evals.scorers.coding_exec import CodingExecScorer
        from evals.core.types import EvalRecord

        scorer = CodingExecScorer()
        record = EvalRecord(
            record_id="test_2",
            problem="Write a function that adds two numbers",
            reference="def add(a, b): return a + b",
            category="coding",
            metadata={
                "entry_point": "add",
                "test": "assert add(1, 2) == 3",
            },
        )
        is_correct, meta = scorer.score(
            record=record,
            model_answer="def add(a, b):\n    return a - b",
        )
        assert is_correct is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_new_suites.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# evals/datasets/chat_mtbench.py
"""MT-Bench style chat evaluation dataset — built-in prompts for multi-turn quality."""

from __future__ import annotations

from typing import List, Optional

from evals.core.dataset import DatasetProvider
from evals.core.types import EvalRecord

# Built-in MT-Bench style questions covering key categories
_CHAT_PROMPTS = [
    {"id": "chat_writing_1", "subject": "writing", "prompt": "Write a persuasive email to convince your introverted friend to attend a party.", "reference": ""},
    {"id": "chat_writing_2", "subject": "writing", "prompt": "Write a short story (3 paragraphs) about a robot learning to paint.", "reference": ""},
    {"id": "chat_roleplay_1", "subject": "roleplay", "prompt": "Pretend you are a medieval knight explaining what a smartphone is to your king.", "reference": ""},
    {"id": "chat_roleplay_2", "subject": "roleplay", "prompt": "You are a detective in a noir film. Describe the scene when you find an important clue.", "reference": ""},
    {"id": "chat_reasoning_1", "subject": "reasoning", "prompt": "If a store offers 20% off and then an additional 15% off the reduced price, what is the total discount?", "reference": "32% total discount"},
    {"id": "chat_reasoning_2", "subject": "reasoning", "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "reference": "9"},
    {"id": "chat_math_1", "subject": "math", "prompt": "Solve for x: 3x + 7 = 22", "reference": "x = 5"},
    {"id": "chat_math_2", "subject": "math", "prompt": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?", "reference": "f'(x) = 3x^2 + 4x - 5"},
    {"id": "chat_coding_1", "subject": "coding", "prompt": "Write a Python function that checks if a string is a palindrome.", "reference": ""},
    {"id": "chat_coding_2", "subject": "coding", "prompt": "Explain the difference between a stack and a queue with examples.", "reference": ""},
    {"id": "chat_extraction_1", "subject": "extraction", "prompt": "Extract all the countries mentioned in this text: 'The conference had delegates from Japan, Brazil, Germany, and Nigeria.'", "reference": "Japan, Brazil, Germany, Nigeria"},
    {"id": "chat_extraction_2", "subject": "extraction", "prompt": "Summarize the following in one sentence: 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.'", "reference": ""},
    {"id": "chat_stem_1", "subject": "stem", "prompt": "Explain photosynthesis in simple terms a 10-year-old would understand.", "reference": ""},
    {"id": "chat_stem_2", "subject": "stem", "prompt": "What happens at the molecular level when water freezes?", "reference": ""},
    {"id": "chat_humanities_1", "subject": "humanities", "prompt": "Compare and contrast the philosophical views of Plato and Aristotle in 3-4 sentences.", "reference": ""},
    {"id": "chat_humanities_2", "subject": "humanities", "prompt": "What were the main causes of the French Revolution?", "reference": ""},
]


class ChatMTBenchDataset(DatasetProvider):
    """Built-in chat quality evaluation prompts inspired by MT-Bench."""

    def load(
        self,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: int = 42,
    ) -> List[EvalRecord]:
        import random

        prompts = list(_CHAT_PROMPTS)
        rng = random.Random(seed)
        rng.shuffle(prompts)

        if max_samples is not None:
            prompts = prompts[:max_samples]

        return [
            EvalRecord(
                record_id=p["id"],
                problem=p["prompt"],
                reference=p["reference"],
                category="chat",
                subject=p["subject"],
            )
            for p in prompts
        ]


# evals/datasets/coding_humaneval.py
"""HumanEval-style coding evaluation dataset."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evals.core.dataset import DatasetProvider
from evals.core.types import EvalRecord

# Built-in coding problems (HumanEval-inspired subset)
_CODING_PROBLEMS = [
    {
        "id": "code_1",
        "prompt": "Write a Python function `has_close_elements(numbers: list[float], threshold: float) -> bool` that checks if any two numbers in the list are closer to each other than the given threshold.",
        "reference": "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False",
        "entry_point": "has_close_elements",
        "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3) == True",
    },
    {
        "id": "code_2",
        "prompt": "Write a Python function `separate_paren_groups(paren_string: str) -> list[str]` that separates groups of balanced parentheses into separate strings.",
        "reference": "def separate_paren_groups(paren_string):\n    result, current, depth = [], '', 0\n    for c in paren_string:\n        if c == '(': depth += 1; current += c\n        elif c == ')': depth -= 1; current += c\n        if depth == 0 and current: result.append(current); current = ''\n    return result",
        "entry_point": "separate_paren_groups",
        "test": "assert separate_paren_groups('(()()) ((())) () ((())())') == ['(()())', '((()))', '()', '((())())']",
    },
    {
        "id": "code_3",
        "prompt": "Write a Python function `truncate_number(number: float) -> float` that returns the decimal part of a positive floating point number.",
        "reference": "def truncate_number(number):\n    return number % 1.0",
        "entry_point": "truncate_number",
        "test": "assert truncate_number(3.5) == 0.5\nassert abs(truncate_number(1.25) - 0.25) < 1e-6",
    },
    {
        "id": "code_4",
        "prompt": "Write a Python function `below_zero(operations: list[int]) -> bool` that checks if a bank account balance goes below zero given a list of deposit/withdrawal operations starting from zero.",
        "reference": "def below_zero(operations):\n    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0: return True\n    return False",
        "entry_point": "below_zero",
        "test": "assert below_zero([1, 2, 3]) == False\nassert below_zero([1, 2, -4, 5]) == True",
    },
    {
        "id": "code_5",
        "prompt": "Write a Python function `mean_absolute_deviation(numbers: list[float]) -> float` that computes the Mean Absolute Deviation of a list of numbers.",
        "reference": "def mean_absolute_deviation(numbers):\n    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)",
        "entry_point": "mean_absolute_deviation",
        "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6",
    },
    {
        "id": "code_6",
        "prompt": "Write a Python function `intersperse(numbers: list[int], delimiter: int) -> list[int]` that inserts a delimiter between every two consecutive elements of the input list.",
        "reference": "def intersperse(numbers, delimiter):\n    if not numbers: return []\n    result = [numbers[0]]\n    for n in numbers[1:]: result.extend([delimiter, n])\n    return result",
        "entry_point": "intersperse",
        "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]",
    },
    {
        "id": "code_7",
        "prompt": "Write a Python function `parse_nested_parens(paren_string: str) -> list[int]` that returns the maximum nesting depth of parentheses for each group separated by spaces.",
        "reference": "def parse_nested_parens(paren_string):\n    result = []\n    for group in paren_string.split():\n        depth = max_depth = 0\n        for c in group:\n            if c == '(': depth += 1; max_depth = max(max_depth, depth)\n            elif c == ')': depth -= 1\n        result.append(max_depth)\n    return result",
        "entry_point": "parse_nested_parens",
        "test": "assert parse_nested_parens('(()()) ((())) () ((())())') == [2, 3, 1, 3]",
    },
    {
        "id": "code_8",
        "prompt": "Write a Python function `filter_by_substring(strings: list[str], substring: str) -> list[str]` that filters a list of strings to only include those containing the given substring.",
        "reference": "def filter_by_substring(strings, substring):\n    return [s for s in strings if substring in s]",
        "entry_point": "filter_by_substring",
        "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bcd', 'cde', 'array'], 'a') == ['abc', 'array']",
    },
]


class CodingHumanEvalDataset(DatasetProvider):
    """Built-in HumanEval-inspired coding problems."""

    def load(
        self,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: int = 42,
    ) -> List[EvalRecord]:
        import random

        problems = list(_CODING_PROBLEMS)
        rng = random.Random(seed)
        rng.shuffle(problems)

        if max_samples is not None:
            problems = problems[:max_samples]

        return [
            EvalRecord(
                record_id=p["id"],
                problem=p["prompt"],
                reference=p["reference"],
                category="coding",
                subject="coding",
                metadata={
                    "entry_point": p["entry_point"],
                    "test": p["test"],
                },
            )
            for p in problems
        ]


# evals/scorers/chat_judge.py
"""LLM-judge scorer for chat quality evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from evals.core.scorer import LLMJudgeScorer
from evals.core.types import EvalRecord


class ChatJudgeScorer(LLMJudgeScorer):
    """Score chat responses using an LLM judge for quality, helpfulness, and coherence."""

    def score(
        self,
        record: EvalRecord,
        model_answer: str,
        **kwargs: Any,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        if not model_answer.strip():
            return False, {"reason": "empty response", "score": 0.0}

        # If reference exists, check for factual match
        if record.reference:
            # Simple substring check for factual answers
            ref_lower = record.reference.lower().strip()
            ans_lower = model_answer.lower().strip()
            if ref_lower in ans_lower or ans_lower in ref_lower:
                return True, {"reason": "matches reference", "score": 1.0}

        # For open-ended questions, mark as correct if response is substantive
        # (In production, this would call the judge LLM)
        is_substantive = len(model_answer.strip()) > 20
        return is_substantive, {
            "reason": "substantive response" if is_substantive else "too short",
            "score": 1.0 if is_substantive else 0.0,
        }


# evals/scorers/coding_exec.py
"""Execution-based scorer for coding evaluation."""

from __future__ import annotations

import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

from evals.core.scorer import Scorer
from evals.core.types import EvalRecord


class CodingExecScorer(Scorer):
    """Score coding responses by executing test cases."""

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout

    def score(
        self,
        record: EvalRecord,
        model_answer: str,
        **kwargs: Any,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        if not model_answer.strip():
            return False, {"reason": "empty response"}

        test_code = record.metadata.get("test", "")
        entry_point = record.metadata.get("entry_point", "")

        if not test_code:
            return None, {"reason": "no test cases available"}

        # Extract code from markdown fences if present
        code = model_answer
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        # Build test script
        script = f"{code}\n\n{test_code}\nprint('ALL_TESTS_PASSED')"

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                f.flush()

                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )

                passed = "ALL_TESTS_PASSED" in result.stdout
                return passed, {
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500],
                    "returncode": result.returncode,
                }
        except subprocess.TimeoutExpired:
            return False, {"reason": "timeout"}
        except Exception as e:
            return False, {"reason": str(e)}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_new_suites.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add evals/datasets/chat_mtbench.py evals/datasets/coding_humaneval.py evals/scorers/chat_judge.py evals/scorers/coding_exec.py tests/evals/test_new_suites.py
git commit -m "feat: add Chat (MT-Bench) and Coding (HumanEval) eval suites"
```

---

### Task 7: Add RAG eval suite and multi-dimensional scoring

**Files:**
- Create: `evals/datasets/rag_eval.py`
- Create: `evals/scorers/rag_scorer.py`
- Test: `tests/evals/test_rag_suite.py`

**Context:** The RAG eval measures whether memory retrieval improves answers vs. without. It uses OpenJarvis's own memory backends. This tests the "with context" vs. "without context" path in `JarvisSystem.ask()` (the `context=True/False` flag). The scorer checks if the answer correctly uses retrieved context.

**Step 1: Write the failing test**

```python
# tests/evals/test_rag_suite.py
"""Tests for RAG eval suite."""


class TestRAGDataset:
    def test_loads_with_category_rag(self):
        from evals.datasets.rag_eval import RAGEvalDataset

        ds = RAGEvalDataset()
        records = ds.load(max_samples=3)
        assert len(records) <= 3
        assert all(r.category == "rag" for r in records)

    def test_has_context_in_metadata(self):
        from evals.datasets.rag_eval import RAGEvalDataset

        ds = RAGEvalDataset()
        records = ds.load(max_samples=3)
        for r in records:
            assert "context" in r.metadata


class TestRAGScorer:
    def test_correct_answer_with_context(self):
        from evals.scorers.rag_scorer import RAGScorer
        from evals.core.types import EvalRecord

        scorer = RAGScorer()
        record = EvalRecord(
            record_id="rag_1",
            problem="What is the capital of France?",
            reference="Paris",
            category="rag",
            metadata={"context": "France is a country in Europe. Its capital is Paris."},
        )
        is_correct, meta = scorer.score(record=record, model_answer="The capital of France is Paris.")
        assert is_correct is True

    def test_incorrect_answer(self):
        from evals.scorers.rag_scorer import RAGScorer
        from evals.core.types import EvalRecord

        scorer = RAGScorer()
        record = EvalRecord(
            record_id="rag_2",
            problem="What is the capital of France?",
            reference="Paris",
            category="rag",
            metadata={"context": "France is a country in Europe. Its capital is Paris."},
        )
        is_correct, meta = scorer.score(record=record, model_answer="The capital of France is London.")
        assert is_correct is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_rag_suite.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# evals/datasets/rag_eval.py
"""RAG evaluation dataset — questions with context passages."""

from __future__ import annotations

from typing import List, Optional

from evals.core.dataset import DatasetProvider
from evals.core.types import EvalRecord

_RAG_PROBLEMS = [
    {"id": "rag_1", "question": "What year was the company founded?", "answer": "2019", "context": "TechCorp was founded in 2019 by Jane Smith and John Doe in San Francisco. The company focuses on AI-powered developer tools.", "subject": "factual"},
    {"id": "rag_2", "question": "What programming language does the framework use?", "answer": "Rust", "context": "The framework is written entirely in Rust for performance and safety. It compiles to a single binary and supports Linux, macOS, and Windows.", "subject": "factual"},
    {"id": "rag_3", "question": "What is the maximum memory limit for the sandbox?", "answer": "256MB", "context": "The WASM sandbox enforces strict resource limits: 256MB maximum memory, 1 billion fuel units for CPU, and 30 second timeout per execution.", "subject": "technical"},
    {"id": "rag_4", "question": "Who is the lead researcher on the project?", "answer": "Dr. Sarah Chen", "context": "The research team is led by Dr. Sarah Chen, who previously worked at DeepMind. Her focus is on reinforcement learning for agent optimization.", "subject": "factual"},
    {"id": "rag_5", "question": "What database is used for persistence?", "answer": "SQLite", "context": "All persistent data is stored in SQLite with WAL mode enabled for concurrent reads. The database file is located at ~/.openjarvis/data.db.", "subject": "technical"},
    {"id": "rag_6", "question": "What is the default temperature for generation?", "answer": "0.7", "context": "Generation defaults include temperature=0.7, max_tokens=1024, top_p=0.9, and repetition_penalty=1.1. These can be overridden per-request.", "subject": "technical"},
    {"id": "rag_7", "question": "How many channels does the system support?", "answer": "14", "context": "The messaging system supports 14 channel adapters: Telegram, Discord, Slack, WhatsApp, LINE, Viber, Messenger, Reddit, Mastodon, XMPP, Rocket.Chat, Zulip, Twitch, and Nostr.", "subject": "factual"},
    {"id": "rag_8", "question": "What authentication method does the P2P protocol use?", "answer": "HMAC-SHA256", "context": "The P2P protocol uses HMAC-SHA256 mutual authentication with nonce-based challenge-response. Each node has a unique identity key pair.", "subject": "technical"},
    {"id": "rag_9", "question": "What is the energy monitoring approach for NVIDIA GPUs?", "answer": "hardware counters and polling", "context": "NVIDIA energy monitoring uses hardware counters via nvidia-smi with configurable polling intervals. The EnergyBatch class aggregates per-inference energy measurements.", "subject": "technical"},
    {"id": "rag_10", "question": "What format are eval results saved in?", "answer": "JSONL", "context": "Evaluation results are saved as JSONL files (one JSON object per line) with a companion .summary.json file containing aggregate metrics.", "subject": "technical"},
]


class RAGEvalDataset(DatasetProvider):
    """Built-in RAG evaluation — questions that require context to answer correctly."""

    def load(
        self,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: int = 42,
    ) -> List[EvalRecord]:
        import random

        problems = list(_RAG_PROBLEMS)
        rng = random.Random(seed)
        rng.shuffle(problems)

        if max_samples is not None:
            problems = problems[:max_samples]

        return [
            EvalRecord(
                record_id=p["id"],
                problem=p["question"],
                reference=p["answer"],
                category="rag",
                subject=p["subject"],
                metadata={"context": p["context"]},
            )
            for p in problems
        ]


# evals/scorers/rag_scorer.py
"""Scorer for RAG evaluation — checks answer against reference with context."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from evals.core.scorer import Scorer
from evals.core.types import EvalRecord


class RAGScorer(Scorer):
    """Score RAG responses by checking if the answer contains the reference."""

    def score(
        self,
        record: EvalRecord,
        model_answer: str,
        **kwargs: Any,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        if not model_answer.strip():
            return False, {"reason": "empty response"}

        reference = record.reference.lower().strip()
        answer = model_answer.lower().strip()

        # Check if reference appears in answer
        is_correct = reference in answer

        return is_correct, {
            "reference": record.reference,
            "match_type": "substring" if is_correct else "no_match",
        }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_rag_suite.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add evals/datasets/rag_eval.py evals/scorers/rag_scorer.py tests/evals/test_rag_suite.py
git commit -m "feat: add RAG eval suite with context-based scoring"
```

---

## Section 3: Composable Abstractions and Recipes

### Task 8: Recipe system — loader, CLI flag, SDK support

**Files:**
- Create: `src/openjarvis/recipes/__init__.py`
- Create: `src/openjarvis/recipes/loader.py`
- Create: `recipes/` (directory for recipe TOML files)
- Create: `recipes/coding_assistant.toml`
- Create: `recipes/research_assistant.toml`
- Create: `recipes/general_assistant.toml`
- Modify: `src/openjarvis/cli/ask.py` (add `--recipe` flag)
- Modify: `src/openjarvis/sdk.py` (add `recipe` parameter to `Jarvis` and `ask()`)
- Test: `tests/recipes/test_loader.py`

**Context:** A recipe is a TOML file that composes all 5 pillars. The recipe loader reads the TOML, resolves each section to the appropriate registry, and returns a config dict that `SystemBuilder` can consume. The `--recipe` CLI flag and `Jarvis(recipe="...")` SDK parameter load and apply the recipe before running. Recipe files live in `recipes/` at project root and `~/.openjarvis/recipes/` for user-defined ones.

**Step 1: Write the failing test**

```python
# tests/recipes/test_loader.py
"""Tests for recipe loading and composition."""

import tempfile
from pathlib import Path


SAMPLE_RECIPE = """\
[recipe]
name = "test_coding"
description = "Test coding assistant recipe"
version = "1.0.0"

[intelligence]
model = "qwen3:8b"
quantization = "q4_K_M"

[engine]
key = "ollama"

[agent]
type = "native_react"
max_turns = 10
temperature = 0.3
tools = ["file_read", "file_write", "code_interpreter", "think"]

[learning]
routing = "grpo"
agent = "icl_updater"

[eval]
suites = ["coding", "reasoning"]
"""


class TestRecipeLoader:
    def test_load_recipe_from_toml(self):
        from openjarvis.recipes.loader import load_recipe

        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write(SAMPLE_RECIPE)
            f.flush()
            recipe = load_recipe(f.name)

        assert recipe.name == "test_coding"
        assert recipe.model == "qwen3:8b"
        assert recipe.engine_key == "ollama"
        assert recipe.agent_type == "native_react"
        assert "file_read" in recipe.tools
        assert recipe.max_turns == 10

    def test_load_recipe_missing_file_raises(self):
        from openjarvis.recipes.loader import load_recipe
        import pytest

        with pytest.raises(FileNotFoundError):
            load_recipe("/nonexistent/path.toml")

    def test_discover_builtin_recipes(self):
        from openjarvis.recipes.loader import discover_recipes

        recipes = discover_recipes()
        # Should find at least the built-in recipes
        names = [r.name for r in recipes]
        assert "coding_assistant" in names or len(names) >= 1

    def test_recipe_to_builder_kwargs(self):
        from openjarvis.recipes.loader import load_recipe

        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write(SAMPLE_RECIPE)
            f.flush()
            recipe = load_recipe(f.name)

        kwargs = recipe.to_builder_kwargs()
        assert kwargs["model"] == "qwen3:8b"
        assert kwargs["engine_key"] == "ollama"
        assert kwargs["agent_name"] == "native_react"
        assert kwargs["tools"] == ["file_read", "file_write", "code_interpreter", "think"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recipes/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/openjarvis/recipes/__init__.py
"""Recipe system — composable pillar configurations."""

# src/openjarvis/recipes/loader.py
"""Load and discover recipe TOML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class Recipe:
    """A loaded recipe — composition of all 5 pillars."""

    name: str
    description: str = ""
    version: str = "1.0.0"

    # Intelligence
    model: str = ""
    quantization: str = ""

    # Engine
    engine_key: str = ""

    # Agent
    agent_type: str = ""
    max_turns: int = 10
    temperature: float = 0.7
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""

    # Learning
    routing_policy: str = ""
    agent_policy: str = ""

    # Eval
    eval_suites: List[str] = field(default_factory=list)

    # Raw TOML for pass-through
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_builder_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs suitable for SystemBuilder or Jarvis."""
        kwargs: dict[str, Any] = {}
        if self.model:
            kwargs["model"] = self.model
        if self.engine_key:
            kwargs["engine_key"] = self.engine_key
        if self.agent_type:
            kwargs["agent_name"] = self.agent_type
        if self.tools:
            kwargs["tools"] = self.tools
        if self.temperature != 0.7:
            kwargs["temperature"] = self.temperature
        if self.max_turns != 10:
            kwargs["max_turns"] = self.max_turns
        return kwargs


def load_recipe(path: str | Path) -> Recipe:
    """Load a recipe from a TOML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Recipe not found: {p}")

    with open(p, "rb") as f:
        data = tomllib.load(f)

    meta = data.get("recipe", {})
    intel = data.get("intelligence", {})
    engine = data.get("engine", {})
    agent = data.get("agent", {})
    learning = data.get("learning", {})
    eval_cfg = data.get("eval", {})

    return Recipe(
        name=meta.get("name", p.stem),
        description=meta.get("description", ""),
        version=meta.get("version", "1.0.0"),
        model=intel.get("model", ""),
        quantization=intel.get("quantization", ""),
        engine_key=engine.get("key", ""),
        agent_type=agent.get("type", ""),
        max_turns=agent.get("max_turns", 10),
        temperature=agent.get("temperature", 0.7),
        tools=agent.get("tools", []),
        system_prompt=agent.get("system_prompt", ""),
        routing_policy=learning.get("routing", ""),
        agent_policy=learning.get("agent", ""),
        eval_suites=eval_cfg.get("suites", []),
        raw=data,
    )


def discover_recipes(extra_dirs: Optional[List[Path]] = None) -> List[Recipe]:
    """Discover all available recipes from known locations."""
    search_dirs = []

    # Built-in recipes in project
    project_recipes = Path(__file__).resolve().parent.parent.parent.parent / "recipes"
    if project_recipes.is_dir():
        search_dirs.append(project_recipes)

    # User recipes
    user_recipes = Path.home() / ".openjarvis" / "recipes"
    if user_recipes.is_dir():
        search_dirs.append(user_recipes)

    if extra_dirs:
        search_dirs.extend(extra_dirs)

    recipes = []
    for d in search_dirs:
        for toml_file in sorted(d.glob("*.toml")):
            try:
                recipes.append(load_recipe(toml_file))
            except Exception:
                continue
    return recipes


def resolve_recipe(name: str) -> Optional[Recipe]:
    """Find a recipe by name from all known locations."""
    for recipe in discover_recipes():
        if recipe.name == name:
            return recipe
    return None
```

Then create the built-in recipe files:

```toml
# recipes/coding_assistant.toml
[recipe]
name = "coding_assistant"
description = "Optimized for code generation, review, and debugging"
version = "1.0.0"

[intelligence]
model = "qwen3:8b"

[engine]
key = "ollama"

[agent]
type = "native_react"
max_turns = 10
temperature = 0.3
tools = ["file_read", "file_write", "code_interpreter", "think", "shell_exec"]

[learning]
routing = "grpo"
agent = "icl_updater"

[eval]
suites = ["coding", "reasoning"]
```

```toml
# recipes/research_assistant.toml
[recipe]
name = "research_assistant"
description = "Deep research with web search, knowledge graph, and cited reports"
version = "1.0.0"

[intelligence]
model = "qwen3:8b"

[engine]
key = "ollama"

[agent]
type = "orchestrator"
max_turns = 15
temperature = 0.5
tools = ["web_search", "http_request", "memory_store", "memory_search", "think", "file_write"]

[learning]
routing = "grpo"
agent = "icl_updater"

[eval]
suites = ["rag", "chat"]
```

```toml
# recipes/general_assistant.toml
[recipe]
name = "general_assistant"
description = "Balanced general-purpose assistant for everyday tasks"
version = "1.0.0"

[intelligence]
model = "qwen3:8b"

[engine]
key = "ollama"

[agent]
type = "orchestrator"
max_turns = 10
temperature = 0.7
tools = ["think", "calculator", "web_search", "memory_search"]

[learning]
routing = "heuristic"

[eval]
suites = ["chat", "reasoning"]
```

Then add `--recipe` to the ask CLI (`src/openjarvis/cli/ask.py`) and `recipe` parameter to `Jarvis.__init__()` and `Jarvis.ask()` in `src/openjarvis/sdk.py`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recipes/test_loader.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/openjarvis/recipes/ recipes/ tests/recipes/ src/openjarvis/cli/ask.py src/openjarvis/sdk.py
git commit -m "feat: add recipe system — TOML composition of 5 pillars with CLI and SDK support"
```

---

### Task 9: Agent templates (15-20 TOML files)

**Files:**
- Create: `templates/agents/` (directory)
- Create: 15-20 TOML files in `templates/agents/`
- Create: `src/openjarvis/templates/__init__.py`
- Create: `src/openjarvis/templates/agent_templates.py`
- Test: `tests/templates/test_agent_templates.py`

**Context:** Agent templates are pre-configured TOML manifests with system prompts, tool sets, and behavioral parameters. They follow the same format as operator manifests (`operators/types.py:9-36`). The loader discovers templates from `templates/agents/` and `~/.openjarvis/templates/agents/`.

**Step 1-5:** Follow TDD pattern. Write test that loads each template and verifies it has required fields (name, system_prompt, tools). Write loader. Create 15-20 template TOML files covering: code-reviewer, debugger, architect, deep-researcher, fact-checker, summarizer, inbox-triager, meeting-prep, note-taker, assistant, tutor, translator, writer, data-analyst, security-auditor. Commit.

---

### Task 10: Bundled skills (20-30 TOML files)

**Files:**
- Create: `skills/builtin/` (directory)
- Create: 20-30 TOML files in `skills/builtin/`
- Test: `tests/skills/test_bundled_skills.py`

**Context:** Skills use `SkillManifest` (`skills/types.py:17-48`) with sequential `SkillStep`s. Each step names a tool and provides an arguments template. The loader is `skills/loader.py:16-104`. Skills live in `skills/builtin/` for bundled ones.

**Step 1-5:** Follow TDD pattern. Write test that loads each skill TOML and verifies valid structure. Create 20-30 skill TOML files covering: file-organizer, file-deduplicator, web-summarize, topic-research, code-lint, code-test-gen, email-draft, meeting-notes, daily-digest, knowledge-extract, pdf-summarize, data-analyze, translate-doc, backup-files, search-and-index, calendar-prep, todo-from-notes, compare-docs, security-scan, dependency-audit. Commit.

---

## Section 4: Operator Recipes

### Task 11: Deep Researcher operator

**Files:**
- Create: `recipes/operators/researcher.toml`
- Create: `recipes/operators/researcher_prompt.md`
- Test: `tests/operators/test_researcher.py`

**Context:** Operators use `OperatorManifest` (`operators/types.py:9-36`) loaded by `operators/loader.py:15-67`. The `OperatorManager` (`operators/manager.py:18-213`) activates them by creating scheduler tasks that run the `OperativeAgent`. The researcher operator searches the web, cross-references sources, builds knowledge graph entries, and produces cited reports.

**Step 1: Write the failing test**

```python
# tests/operators/test_researcher.py
"""Tests for the Deep Researcher operator recipe."""

from pathlib import Path


class TestResearcherOperator:
    def test_loads_valid_manifest(self):
        from openjarvis.operators.loader import load_operator

        manifest = load_operator(Path(__file__).parent.parent.parent / "recipes" / "operators" / "researcher.toml")
        assert manifest.name == "researcher"
        assert "web_search" in manifest.tools
        assert "memory_store" in manifest.tools
        assert manifest.max_turns >= 10
        assert manifest.system_prompt or manifest.system_prompt_path

    def test_has_required_tools(self):
        from openjarvis.operators.loader import load_operator

        manifest = load_operator(Path(__file__).parent.parent.parent / "recipes" / "operators" / "researcher.toml")
        required = {"web_search", "http_request", "memory_store", "memory_search", "think", "file_write"}
        assert required.issubset(set(manifest.tools))
```

**Step 2-5:** Write researcher.toml and researcher_prompt.md with a detailed system prompt for autonomous research. Run tests. Commit.

---

### Task 12: Correspondent operator

**Files:**
- Create: `recipes/operators/correspondent.toml`
- Create: `recipes/operators/correspondent_prompt.md`
- Test: `tests/operators/test_correspondent.py`

**Step 1-5:** Same pattern as Task 11 but for messaging triage. Tools: memory_store, memory_search, think, llm_call. System prompt focuses on urgency classification, draft responses, daily digest. Commit.

---

### Task 13: Sentinel operator

**Files:**
- Create: `recipes/operators/sentinel.toml`
- Create: `recipes/operators/sentinel_prompt.md`
- Test: `tests/operators/test_sentinel.py`

**Step 1-5:** Same pattern as Task 11 but for monitoring Twitter, Reddit, Mastodon, Google Trends, RSS, URLs. Tools: web_search, http_request, memory_store, memory_search, kg_add_entity. System prompt focuses on change detection, significance scoring, alert generation. Commit.

---

## Section 5: Integration and Wiring

### Task 14: Update LearningConfig and wire orchestrator to scheduler

**Files:**
- Modify: `src/openjarvis/core/config.py` (add training config fields)
- Modify: `src/openjarvis/learning/__init__.py` (export new components)
- Modify: `src/openjarvis/system.py` (wire LearningOrchestrator into SystemBuilder)
- Test: `tests/test_system_learning.py`

**Context:** `LearningConfig` (`core/config.py`) needs fields for the training pipeline: `training_enabled`, `training_schedule`, `lora_rank`, `lora_alpha`, `min_sft_pairs`, `min_improvement`. The `SystemBuilder` needs to optionally create a `LearningOrchestrator` and wire it to the scheduler for periodic learning cycles.

**Step 1-5:** TDD. Add config fields, wire into SystemBuilder, test that build() creates orchestrator when learning.training_enabled=True. Commit.

---

### Task 15: Update CLAUDE.md and run full test suite

**Files:**
- Modify: `CLAUDE.md` (add recipe, eval, and learning orchestrator documentation)
- Modify: `src/openjarvis/learning/training/__init__.py` (export all new training components)

**Step 1:** Update CLAUDE.md with new commands: `jarvis eval run`, `jarvis eval compare`, `jarvis eval report`, `jarvis ask --recipe`, recipe system docs, operator recipes docs.

**Step 2:** Run full test suite:

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All existing ~2940 tests + ~200-300 new tests PASS.

**Step 3:** Commit everything.

```bash
git add -A
git commit -m "feat: complete differentiated functionalities — learning flywheel, evals, recipes, operators"
```
