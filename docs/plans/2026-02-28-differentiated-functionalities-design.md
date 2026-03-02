# Differentiated Functionalities Design

**Date:** 2026-02-28
**Approach:** Learning Flywheel (Approach A)
**Status:** Approved

## Vision

Make OpenJarvis's functionalities genuinely differentiated through four pillars of uniqueness:

1. **Fully open-source on-device stack** — every component runs locally, user controls data
2. **Native on-device intelligence** — make local LMs punch above their weight via trace-driven learning
3. **Latency, privacy, energy as first-class** — alongside accuracy, not afterthoughts
4. **Composable, programmable abstractions** — Intelligence, Agent, Tools, Engine, Learning as optimized components

**Priority:** On-device depth first. Trace-driven learning is the core differentiator.

## Design Decisions

- **Approach A selected:** Build the trace-to-learn-to-eval loop as the backbone, hang user-facing features off it
- **Learning targets:** Full stack — routing, agent logic, AND model weights (LoRA/QLoRA)
- **Eval purpose:** Research platform — rigorous before/after measurement with reproducible configs
- **Composition layers:** Python SDK + TOML configs + pre-built recipes
- **Operators:** 3 recipes (researcher, correspondent, sentinel) — not a new abstraction, just recipe + schedule + channel
- **Interleave:** Wire up real learning AND build new user-facing features simultaneously

## Section 1: Trace-Driven Learning Pipeline

The core differentiator. Makes the learning pipeline real, not just infrastructure.

### Current State

`TraceStore` captures traces, `TraceAnalyzer` computes stats, GRPO/Bandit policies update internal routing state, `SFTRouterPolicy` and `AgentAdvisorPolicy` exist but don't produce real training jobs, `ICLUpdaterPolicy` updates in-context examples with versioning/rollback.

### New Components

**Trace-to-Training Data pipeline** (`learning/training/data.py`): Mines the `TraceStore` for supervised pairs. For successful traces (user rated positively or task completed), extracts `(input, preferred_output)` pairs. For routing, extracts `(query_class, best_model)` pairs. For agent logic, extracts `(query_type, best_agent_config)` tuples. Filters: minimum trace quality score, deduplication, privacy scanning (strip PII before training).

**LoRA/QLoRA fine-tuning** (`learning/training/lora.py`): Wraps `transformers` + `peft` + `trl` for actual weight updates on local models. Takes training data from the pipeline, runs LoRA fine-tuning with configurable rank/alpha/dropout. Produces adapter weights saved alongside the base model. Energy-aware via existing `EnergyMonitor`. Guarded by hardware detection — only runs if sufficient VRAM/RAM.

**Agent config evolution** (`learning/agent_learning.py`): Makes `AgentAdvisorPolicy` actually rewrite agent TOML configs. Analyzes traces to identify: which tools were useful vs. never called, which system prompt patterns led to success, optimal `max_turns` per query class. Writes updated configs with version control (git-tracked, rollback via `ICLUpdaterPolicy` pattern).

**Learning orchestrator** (`learning/orchestrator.py`): Coordinates all learning. Runs on a schedule (e.g., nightly) or on-demand. Steps: collect new traces, mine training data, run evals (baseline), fine-tune model / update agent configs, run evals (after), accept/reject based on improvement threshold. Atomic: if evals don't improve, rollback automatically.

### Data Flow

```
User queries -> Traces recorded -> TraceStore (SQLite)
                                       |
               LearningOrchestrator (scheduled/on-demand)
                     |                    |                    |
           TrainingDataMiner     AgentConfigEvolver    RoutingPolicyUpdater
                     |                    |                    |
           LoRA fine-tune         TOML config update     GRPO/Bandit update
                     |                    |                    |
               EvalHarness (before/after comparison)
                     |
           Accept (deploy) or Reject (rollback)
```

## Section 2: Eval Framework

Proves the learning flywheel works and serves as the research platform.

### Five Workload-Type Eval Suites

- **Chat:** Multi-turn conversation quality (MT-Bench style, coherence, helpfulness)
- **Reasoning:** Logic/math/science (GSM8K, ARC, MMLU subsets that run locally)
- **RAG:** Retrieval-augmented accuracy (measures whether memory retrieval improves answers vs. without, uses OpenJarvis's own memory backends)
- **Agentic:** Task completion rate for multi-step tool-use workflows (custom scenarios: research a topic, triage inbox, etc.)
- **Coding:** Code generation correctness (HumanEval, MBPP subsets)

### On-Device Metrics as First-Class

Every eval records: accuracy, latency (TTFT + total), energy consumption (via `EnergyMonitor`), peak memory, tokens/second. Results are multi-dimensional, not a single number. Configurable weighting (e.g., 40% accuracy, 30% latency, 20% energy, 10% memory).

### Before/After Measurement

`LearningOrchestrator` calls the eval harness before learning (baseline) and after (candidate). Improvement measured per-dimension. Configurable acceptance threshold (e.g., accept if accuracy improves >2% without latency regressing >10%).

### Reproducible Configs

Every eval run fully specified by TOML (model, agent, tools, dataset, metrics, hardware). Results are JSONL with full provenance. CLI: `jarvis eval run`, `jarvis eval compare`, `jarvis eval report`.

### Ablation Support

Run the same eval with one variable changed (with vs. without LoRA, with vs. without a tool, with vs. without memory). Built-in diffing and statistical significance testing.

## Section 3: Composable Abstractions and Recipes

The user-facing layer for interacting with the pillars.

### Recipe System (`recipes/`)

A recipe is a curated composition of all 5 pillars in a single TOML file: model selection, engine preference, agent type + system prompt, tool set, learning policy, and eval config. Recipes are opinionated defaults that work well together.

Users load recipes via `jarvis ask --recipe coding_assistant "Fix this bug"` or `Jarvis(recipe="coding_assistant")` in Python.

### Agent Templates (15-20)

Pre-configured agent TOML manifests with system prompts, tool sets, and behavioral parameters.

Categories:
- **Coding:** code-reviewer, debugger, architect
- **Research:** deep-researcher, fact-checker, summarizer
- **Productivity:** inbox-triager, meeting-prep, note-taker
- **General:** assistant, tutor, translator, writer

### Bundled Skills (20-30)

Focused on on-device use cases where local execution matters.

Categories:
- **File management:** organize, deduplicate, backup
- **Personal knowledge:** extract, summarize, index
- **Development:** lint, test, review
- **Productivity:** calendar prep, email draft, todo management

### Three Composition Layers

- **CLI flags:** `--recipe`, `--agent`, `--tools`, `--model` for quick use
- **TOML configs:** Full pillar configuration for persistent setups
- **Python SDK:** `SystemBuilder` for programmatic composition, research scripts, custom pipelines

## Section 4: Operator Recipes

OpenJarvis's answer to autonomous task agents. Operators are NOT a new abstraction — they are recipe + schedule + channel output, composed via TOML.

### Deep Researcher (`recipes/operators/researcher.toml`)

Autonomous research agent. Given a topic, searches the web, cross-references sources, evaluates credibility, builds a knowledge graph entry, produces a cited report. Runs on-demand or scheduled.

Tools: `web_search`, `http_request`, `memory_store`, `memory_search`, `kg_add_entity`, `kg_add_relation`, `file_write`.

Learning: traces which search strategies yield useful results, evolves search query patterns over time.

### Correspondent (`recipes/operators/correspondent.toml`)

Messaging triage across Slack/Gmail/Discord. Monitors incoming messages, classifies urgency (urgent/normal/low/ignore), drafts responses for high-priority items, summarizes the rest into a daily digest.

Tools: `memory_store`, `memory_search`, `think`, `llm_call`.

Channels: Slack, email, Discord.

Learning: adapts to user's triage preferences over time.

### Sentinel (`recipes/operators/sentinel.toml`)

Monitors Twitter, Reddit, Mastodon, Google Trends, RSS feeds, and specified URLs for changes relevant to user-defined topics. Produces alerts when something significant surfaces — trending discussions, sentiment shifts, breaking news, competitor activity.

Tools: `web_search`, `http_request`, `memory_store`, `memory_search`, `kg_add_entity`.

Channels: Twitter, Reddit, Mastodon (read via APIs), Google Trends.

Learning: refines what "significant" means based on which alerts the user acts on vs. dismisses.

## Section 5: Testing and Validation

### Learning Pipeline Tests

Unit tests for `TrainingDataMiner` (mock traces, verify correct pairs), `LoRATrainer` (mock training, verify adapter saved), `AgentConfigEvolver` (mock traces, verify TOML rewritten), `LearningOrchestrator` (mock all steps, verify accept/reject logic). Integration test: synthetic traces, actual LoRA training on tiny model, verify loss decreases.

### Eval Framework Tests

Each of the 5 eval suites has a smoke mode (5-10 examples, <30 seconds). Verify multi-dimensional scoring. Verify before/after comparison and acceptance thresholds. Verify ablation diffing.

### Recipe and Template Tests

Each recipe loads without error. Each agent template produces a valid agent. Each skill executes its steps. Composition tests: `--recipe` flag wires all pillars correctly.

### Operator Tests

Each operator recipe loads and schedules correctly. Mock tool execution to verify agent loop completes. Verify learning feedback loop (trace, config update, re-eval).

### Regression

All ~2940 existing tests continue to pass. Target ~200-300 additional tests for new functionality.
