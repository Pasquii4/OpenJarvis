# Silent Exception Remediation Design

**Date:** 2026-03-09
**Status:** Approved
**Branch:** `feat/feature-gap-closure`

## Problem

The codebase contains ~333 `except` blocks that suppress exceptions. Of these:

- ~188 are **import guards** (`except ImportError: pass`) — intentional, no change needed.
- ~20-30 are **intentionally silent with comments** — acceptable but should get `logger.debug()`.
- ~30-40 are **silent pass blocks without justification** — need `logger.warning()`.
- ~52 are **return/continue without logging** — need `logger.warning()` before the return.

**Total blocks requiring remediation: ~90-100.**

In a system orchestrating multiple engines, agents, tools, and learning policies, silent failures make debugging extremely difficult. The system degrades quietly instead of surfacing errors.

## Design Decisions

### Logging level policy

| Category | Level | Rationale |
|----------|-------|-----------|
| Problematic silent blocks (no comment, no logging) | `logger.warning()` | Visible by default; makes failures observable |
| Intentionally silent blocks (best-effort, with comments) | `logger.debug()` | Traceable when needed, not noisy by default |
| Import guards (`except ImportError`) | No change | Intentional project pattern for optional deps |

### Per-file setup

Each modified file must have at the top (if not already present):

```python
import logging

logger = logging.getLogger(__name__)
```

### Transformation patterns

**Silent pass (no comment):**
```python
# Before:
except Exception:
    pass

# After:
except Exception as exc:
    logger.warning("Failed to <context>: %s", exc)
```

**Return without logging:**
```python
# Before:
except (SomeError, OtherError):
    return []

# After:
except (SomeError, OtherError) as exc:
    logger.warning("Failed to <context>: %s", exc)
    return []
```

**Intentionally silent (best-effort):**
```python
# Before:
except Exception:
    pass  # telemetry is best-effort

# After:
except Exception as exc:
    logger.debug("Best-effort <context> failed: %s", exc)  # non-fatal
```

### What does NOT change

- Import guards (`except ImportError: pass`) are untouched.
- Runtime behavior: no new exceptions raised, no control flow changes.
- Log format: `%s` style (not f-strings) per Python logging best practice.

## Phased Execution Order

### Phase 1 — Critical path (core system + public API)
1. `src/openjarvis/system.py` (8 blocks)
2. `src/openjarvis/sdk.py` (10 blocks)
3. `src/openjarvis/engine/ollama.py` + `_openai_compat.py` (inference path)

### Phase 2 — User-facing (CLI + engine discovery)
4. `src/openjarvis/cli/ask.py` (7 blocks)
5. `src/openjarvis/cli/quickstart_cmd.py`
6. `src/openjarvis/engine/_discovery.py`

### Phase 3 — Agent + tools (runtime logic)
7. `src/openjarvis/agents/monitor_operative.py`
8. `src/openjarvis/tools/` (any silent blocks)
9. `src/openjarvis/security/`

### Phase 4 — Supporting systems (telemetry, learning, remaining)
10. `src/openjarvis/telemetry/energy_*.py`
11. `src/openjarvis/learning/`
12. Any remaining files

## Verification

- `uv run ruff check src/ tests/` passes (zero lint warnings).
- `uv run pytest tests/ -v --tb=short` passes (no behavioral changes).
- Manual grep confirms no remaining bare `except ... pass` without logging or comment (excluding import guards).
