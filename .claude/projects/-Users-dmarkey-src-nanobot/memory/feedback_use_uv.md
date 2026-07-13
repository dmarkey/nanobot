---
name: Use uv for Python commands
description: User prefers uv run for running Python/nanobot commands instead of bare python
type: feedback
---

Use `uv run` when running Python commands in the nanobot project, not bare `python`.

**Why:** User explicitly corrected bare `python -c` to use `uv` instead — project uses uv for dependency management.

**How to apply:** Always prefix Python invocations with `uv run` (e.g., `uv run python -c ...`, `uv run pytest ...`).
