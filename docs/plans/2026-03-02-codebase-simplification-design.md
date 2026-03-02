# Codebase Simplification Design

**Date:** 2026-03-02
**Goal:** Simplify repository structure and clean up code while preserving all functionality.
**Approach:** Incremental commits, tests verified after each step.

## 1. Target Root Structure

Reduce root from 32 items to 21 items.

### Items removed from git tracking
- `get-pip.py` (2.2MB, regenerable)
- `site/` (9.6MB, mkdocs build output)
- `results/` (5.7MB, eval output)

### Items moved
| Source | Destination |
|--------|------------|
| `Dockerfile`, `Dockerfile.gpu`, `Dockerfile.gpu.rocm`, `Dockerfile.sandbox` | `deploy/docker/` |
| `docker-compose.yml`, `docker-compose.gpu.rocm.yml` | `deploy/docker/` |
| `recipes/*.toml` | `src/openjarvis/recipes/data/` |
| `templates/agents/*.toml` | `src/openjarvis/templates/data/` |
| `skills/builtin/*.toml` | `src/openjarvis/skills/data/` |
| `operators/*.toml` | `src/openjarvis/operators/data/` |
| `evals/` (entire directory) | `src/openjarvis/evals/` |

### Items kept at root
- `CLAUDE.md`, `README.md`, `VISION.md`, `ROADMAP.md`, `NOTES.md`
- `mkdocs.yml`, `pyproject.toml`, `uv.lock`, `LICENSE`
- `assets/`, `configs/`, `deploy/`, `desktop/`, `docs/`, `frontend/`, `scripts/`, `src/`, `tests/`

## 2. Data Directory Migration

Each module's `discover_*()` function defaults to `Path(__file__).parent / "data"` for built-in TOML files. `pyproject.toml` declares package data via `[tool.hatch.build.targets.wheel]`.

### evals/ migration
The entire evals directory (Python modules + configs + dataset loaders) moves into `src/openjarvis/evals/` as a subpackage. CLI `jarvis eval` and `python -m openjarvis.evals` entry points updated.

## 3. Code Cleanups

### 3a. Remove memory/ shim package
Delete `src/openjarvis/memory/` (11 files, 108 lines of pure re-exports). Update all imports to use `openjarvis.tools.storage.*` directly.

### 3b. Consolidate engine wrappers
Replace 5 near-identical 17-line files (vllm.py, sglang.py, llamacpp.py, mlx.py, lmstudio.py) with a single data-driven registration in `openai_compat_engines.py`.

### 3c. Refactor repetitive __init__.py imports
Replace 26 identical try-except blocks in channels/__init__.py (and similar in learning/, engine/) with loop-based auto-imports using `importlib.import_module()`.

## 4. Test & CI Impact

- Update tests importing from `openjarvis.memory.*` to `openjarvis.tools.storage.*`
- Delete `tests/memory/` (duplicates `tests/tools/` storage tests)
- Update engine tests for consolidated module
- Update CI workflows referencing moved paths
- Update `pyproject.toml` entry points
- All ~3240 tests must pass after each commit

## 5. Commit Sequence

1. Clean git artifacts (get-pip.py, site/, results/) + update .gitignore
2. Move Docker files to deploy/docker/
3. Move recipes/ TOML data into src/openjarvis/recipes/data/
4. Move templates/ TOML data into src/openjarvis/templates/data/
5. Move skills/ TOML data into src/openjarvis/skills/data/
6. Move operators/ TOML data into src/openjarvis/operators/data/
7. Move evals/ into src/openjarvis/evals/
8. Remove memory/ shim package + update imports
9. Consolidate engine wrappers into single file
10. Refactor repetitive __init__.py imports
11. Update CLAUDE.md to reflect new structure
