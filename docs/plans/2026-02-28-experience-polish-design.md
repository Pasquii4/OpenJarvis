# Experience Polish Design: Eval, CLI, Install, Dashboard

**Date**: 2026-02-28
**Status**: Approved
**Approach**: Vertical slices (3 mini-phases, independently shippable)

## Context

OpenJarvis has strong infrastructure (multi-vendor energy monitoring, eval framework, Tauri desktop, PWA) but the end-to-end user experience has gaps. This design addresses: eval output quality, CLI polish, installation flow, and dashboard aesthetics.

### Current State (from audits)

| Area | Score | Key Gaps |
|------|-------|----------|
| Energy Monitors | 8.5/10 | IPJ/IPW only in evals, not core telemetry |
| Eval Framework | 9/10 | Token stats null, no grouped tables, no trace aggregation |
| CLI Formatting | 9/10 | No logging, no verbose/quiet, bench lacks full stats |
| Installation | 8.5/10 | No quickstart, no auto-suggestions in errors |
| Desktop App | 8/10 | No settings panel, placeholder icon, stub tray |
| Browser/PWA | 6/10 | No CI/CD, no versioning, no error boundary |

### Key Decisions

- **Benchmark focus**: GAIA with OpenHands (TerminalBench deferred)
- **IPW** = task_accuracy / avg_power_watts (task-level, NOT per-inference)
- **IPJ** = task_accuracy / avg_energy_joules (task-level, NOT per-inference)
- **tokens_per_joule** = per-inference efficiency metric in core telemetry
- **Table layout**: Grouped panels by default, `--compact` for single dense table
- **Trace detail**: Summary + step-type breakdown by default, `--trace-detail` for full listing
- **Installation**: Non-interactive `jarvis quickstart` (zero prompts)
- **Dashboard style**: Clean, minimalistic (ChatGPT/Claude aesthetic) with unique side panels

---

## Phase 23a: Perfect Eval Run

Everything needed so `python -m evals run -c config.toml` produces beautiful, complete results.

### 1. Core Telemetry: tokens_per_joule

**Files**:
- `src/openjarvis/core/types.py` — Add `tokens_per_joule` to `TelemetryRecord`
- `src/openjarvis/telemetry/instrumented_engine.py` — Compute `tokens_per_joule = completion_tokens / energy_joules`
- `src/openjarvis/telemetry/store.py` — Schema migration for new column
- `src/openjarvis/telemetry/aggregator.py` — Add `avg_tokens_per_joule` to `ModelStats`/`EngineStats`

**Not** adding IPJ/IPW to core telemetry — they require task accuracy and are inherently eval-level.

### 2. Eval Metrics: Fix Token Stats + Strengthen IPJ/IPW

**Files**:
- `evals/core/runner.py`:
  - Wire `prompt_tokens` and `completion_tokens` from engine response into `EvalResult`
  - Verify IPW = `accuracy / avg(power_watts)` across all scored samples
  - Verify IPJ = `accuracy / avg(energy_joules)` across all scored samples
  - Ensure `energy_joules` and `power_watts` populated from telemetry for every sample
- `evals/core/types.py`:
  - Add `total_energy_joules` (sum over all samples) to `RunSummary`
  - Add `avg_power_watts` (mean over all samples) to `RunSummary`
  - Add `trace_steps: int`, `trace_energy_joules: float` to `EvalResult`
  - Add `trace_step_type_stats: Dict[str, StepTypeStats]` to `RunSummary`

### 3. Eval Display: Grouped Tables

Rewrite `evals/core/display.py` with these functions:

#### `print_accuracy_panel(summary)`
```
╭─ Accuracy ──────────────────────────────────────────────────╮
│ Overall Accuracy    42.0%  (84/200)                         │
│ Level 1             58.8%  (40/68)                          │
│ Level 2             35.0%  (35/100)                         │
│ Level 3             28.1%  (9/32)                           │
╰─────────────────────────────────────────────────────────────╯
```

#### `print_latency_table(summary)`
```
╭─ Latency & Throughput ──────────────────────────────────────╮
│ Metric                 Avg     Median   Min     Max    Std  │
│ Latency (s)           15.41   12.30    2.10   89.50  11.20 │
│ TTFT (ms)            145.2   120.0    45.0   890.0   85.3  │
│ Throughput (tok/s)    41.9    38.5    12.0    95.0   15.2  │
│ Avg Input Tokens    1024     890      128    4096    520   │
│ Avg Output Tokens    256     210       32    2048    180   │
╰─────────────────────────────────────────────────────────────╯
```

#### `print_energy_table(summary)`
```
╭─ Energy & Efficiency ───────────────────────────────────────╮
│ Metric                 Avg     Median   Min     Max    Std  │
│ Energy (J)          46502   38500    4145  410522  89390   │
│ Power (W)           883.8   870.2   650.0  1050.0   85.0  │
│ GPU Util (%)         46.4    48.0    12.0    92.0   18.5  │
│ Energy/Token (mJ)    12.5    10.8     3.2    45.0    8.1  │
│ Tokens/Joule         80.0    92.6    22.2   312.5   65.0  │
│ ──────────────────────────────────────────────────────────  │
│ IPW (acc/W)         0.00048                                 │
│ IPJ (acc/J)         9.03e-06                                │
│ Total Energy (kJ)   9300.4                                  │
╰─────────────────────────────────────────────────────────────╯
```

#### `print_trace_summary(summary)`
```
╭─ Agentic Trace Summary ────────────────────────────────────╮
│ Total Steps: 1240  │  Avg Steps/Sample: 6.2                │
│                                                             │
│ Step Type      Count  Avg Duration  Avg Energy  Avg In Tok  Avg Out Tok  │
│                              Median/Min/Max/Std for each column          │
│ generate         580    8.2s        38200 J       890        256         │
│ tool_call        420    3.1s          —            —          —          │
│ retrieve         120    0.8s          —            —          —          │
│ route            120    0.01s         —            —          —          │
╰─────────────────────────────────────────────────────────────╯
```

All metrics in trace summary show avg/median/min/max/std where applicable.

#### `print_compact_table(summary)` — `--compact` flag
Single dense table with all 17 metrics as rows, columns: avg/median/min/max/std.

#### CLI flags
- `--compact`: Dense single table
- `--trace-detail`: Full per-step listing for each sample

### 4. Per-Step Trace Aggregation

**Files**:
- `src/openjarvis/traces/analyzer.py`:
  - Add `TraceSummary.total_energy_joules` — sum of `step.metadata['energy_joules']`
  - Add `TraceSummary.total_generate_energy_joules` — sum for GENERATE steps
  - Add `TraceSummary.step_type_stats` — dict mapping step type to `{count, avg_duration, median_duration, min_duration, max_duration, std_duration, total_energy, avg_input_tokens, median_input_tokens, min_input_tokens, max_input_tokens, std_input_tokens, avg_output_tokens, median_output_tokens, min_output_tokens, max_output_tokens, std_output_tokens}`
- `evals/core/runner.py` — After each agent eval sample, extract `TraceSummary` and store in `EvalResult.metadata`

---

## Phase 23b: Perfect First Experience

Everything needed so a new user goes from zero to first eval in one sitting.

### 5. `jarvis quickstart` Command

**New file**: `src/openjarvis/cli/quickstart_cmd.py`

Non-interactive flow (5 numbered steps):
1. Detect hardware (platform, CPU, RAM, GPU vendor/model/VRAM)
2. Write config to `~/.openjarvis/config.toml` (skip if exists, unless `--force`)
3. Check engine health (try auto-detected engine)
4. Verify model availability (list models from engine)
5. Test query ("What is 2+2?") with latency + energy measurement

Each step prints a status line. If a step fails, print a helpful suggestion and exit gracefully.

Flags: `--force` (redo everything)

**Register** in `src/openjarvis/cli/__init__.py`.

### 6. Error Message Auto-Suggestions

**New file**: `src/openjarvis/cli/hints.py`

Centralized hint functions:
- `hint_no_config()` → "Config not found. Run: `jarvis quickstart`"
- `hint_no_engine()` → "No engine responding. Run: `jarvis doctor`"
- `hint_no_model()` → "No model available. Try: `ollama pull qwen3:8b`"

**Wire into**: `ask.py`, `serve.py`, `bench_cmd.py`, `chat_cmd.py` at failure points.

### 7. Global Logging & Verbose/Quiet Flags

**Changes**:
- `src/openjarvis/cli/__init__.py` — Add `--verbose` / `--quiet` to root `cli` group
- **New file**: `src/openjarvis/cli/log_config.py` — Centralized logging setup
  - `RichHandler` for console (respects quiet flag)
  - `RotatingFileHandler` for `~/.openjarvis/cli.log` (5 MB max, 3 backups)
  - Default level: WARNING; `--verbose`: DEBUG; `--quiet`: ERROR

**Add progress indicators**:
- `ask.py` — Spinner during generation
- `memory_cmd.py` — Progress bar for indexing

### 8. Bench CLI Full Stats Tables

**Changes**:
- `src/openjarvis/cli/bench_cmd.py` — Show full stats table (avg/median/min/max/std) matching eval style
- `src/openjarvis/bench/latency.py` — Export all percentile stats (already computes internally)
- `src/openjarvis/bench/throughput.py` — Add stats computation
- `src/openjarvis/bench/energy.py` — Add stats computation

---

## Phase 23c: Perfect Dashboard

Desktop and browser apps polished for demos and daily use.

### Design Aesthetic

**Clean, minimalistic** — inspired by ChatGPT and Claude web interfaces:
- Generous whitespace, muted color palette, subtle borders
- Sans-serif typography, comfortable line heights
- Smooth transitions, no visual clutter

**Differentiated** via unique side panels:
- Energy dashboard with real-time power charts
- Trace debugger with color-coded timeline
- Learning curve visualization
- Memory browser with relevance scoring

The side panels are what make OpenJarvis unique vs generic chat UIs. Keep them accessible but not overwhelming — collapsible, clean data density.

### 9. Desktop: Settings Panel

**New file**: `desktop/src/components/SettingsPanel.tsx`
- API URL configuration (default: `localhost:8000`)
- Auto-update interval setting
- Theme toggle (dark/light) — currently hardcoded dark
- Persist to `localStorage`

### 10. Desktop: Windows Icon

Replace 108-byte placeholder `desktop/src-tauri/icons/icon.ico` with proper multi-resolution icon generated from existing `icon.png`.

### 11. Desktop: System Tray

Implement stubbed tray menu in `desktop/src-tauri/src/lib.rs`:
- Show/Hide window toggle
- Health status indicator (green/red dot)
- Quit action

### 12. Browser/PWA: CI/CD

**New file**: `.github/workflows/frontend.yml`
- Trigger on changes to `frontend/`
- `npm ci && npm run build`
- Commit built output to `src/openjarvis/server/static/`
- Add `version` field to `manifest.webmanifest` (auto-bumped)

### 13. Browser/PWA: Error Boundary + Config

- Wrap `<App />` in React error boundary (graceful crash recovery)
- Support `VITE_API_URL` environment variable (default `localhost:8000`)

### 14. Browser/PWA: Style Refresh

Refactor the 12,971-line `App.css`:
- Modularize into component-scoped CSS files
- Clean minimalist aesthetic (ChatGPT/Claude-inspired)
- Catppuccin-based dark theme (keep existing) + light theme option
- Collapsible side panels with smooth transitions

---

## Implementation Order

### Phase 23a (highest priority — enables GAIA research)
1. Core telemetry: `tokens_per_joule`
2. Eval metrics: fix token stats, strengthen IPJ/IPW
3. Per-step trace aggregation in `TraceAnalyzer`
4. Eval display: grouped tables, `--compact`, `--trace-detail`

### Phase 23b (high priority — onboarding & CLI)
5. `jarvis quickstart` command
6. Error message auto-suggestions
7. Global logging + verbose/quiet flags
8. Bench CLI full stats tables

### Phase 23c (lower priority — dashboard polish)
9. Desktop settings panel
10. Windows icon fix
11. System tray implementation
12. PWA CI/CD
13. PWA error boundary + config
14. PWA style refresh

---

## Files Modified (Summary)

### Phase 23a
| File | Change |
|------|--------|
| `src/openjarvis/core/types.py` | Add `tokens_per_joule` to `TelemetryRecord` |
| `src/openjarvis/telemetry/instrumented_engine.py` | Compute `tokens_per_joule` |
| `src/openjarvis/telemetry/store.py` | Schema migration |
| `src/openjarvis/telemetry/aggregator.py` | Add `avg_tokens_per_joule` |
| `src/openjarvis/traces/analyzer.py` | Add energy aggregation, step-type stats |
| `evals/core/types.py` | Add trace fields, total_energy, avg_power |
| `evals/core/runner.py` | Fix token stats, wire trace data, verify IPJ/IPW |
| `evals/core/display.py` | Rewrite: grouped panels, compact mode |
| `evals/cli.py` | Add `--compact`, `--trace-detail` flags |

### Phase 23b
| File | Change |
|------|--------|
| `src/openjarvis/cli/quickstart_cmd.py` | **New**: quickstart command |
| `src/openjarvis/cli/hints.py` | **New**: centralized hint system |
| `src/openjarvis/cli/log_config.py` | **New**: logging setup |
| `src/openjarvis/cli/__init__.py` | Register quickstart, add verbose/quiet |
| `src/openjarvis/cli/ask.py` | Add hints, progress spinner |
| `src/openjarvis/cli/serve.py` | Add hints |
| `src/openjarvis/cli/bench_cmd.py` | Full stats tables |
| `src/openjarvis/cli/chat_cmd.py` | Add hints |
| `src/openjarvis/cli/memory_cmd.py` | Progress bar for indexing |
| `src/openjarvis/bench/latency.py` | Export full stats |
| `src/openjarvis/bench/throughput.py` | Add stats computation |
| `src/openjarvis/bench/energy.py` | Add stats computation |

### Phase 23c
| File | Change |
|------|--------|
| `desktop/src/components/SettingsPanel.tsx` | **New**: settings UI |
| `desktop/src-tauri/icons/icon.ico` | Replace placeholder |
| `desktop/src-tauri/src/lib.rs` | Implement system tray |
| `.github/workflows/frontend.yml` | **New**: PWA CI/CD |
| `frontend/src/App.tsx` | Error boundary wrapper |
| `frontend/vite.config.ts` | VITE_API_URL support |
| `frontend/src/App.css` | Modularize, style refresh |
