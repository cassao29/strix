"""Ambient handle to the active run's context dict.

Strix's Runner passes a per-run ``context`` dict (coordinator, ``agent_id``,
``parent_id``, budgets, ...) into ``Runner.run_streamed(context=...)``, and
tools read it through the ``ToolContext`` the Runner builds for each call.

The ``Model`` interface, however, receives none of that — it is constructed
from a model-name string and its ``get_response`` gets no run context. The
claude-code backend needs it: it re-exposes Strix's tools to Claude Code over
MCP and invokes them itself, so it must reconstruct a faithful
``ToolContext`` (otherwise coordination tools that read ``context["coordinator"]``
no-op). This ContextVar bridges the gap: ``execution.py`` sets it right before
``Runner.run_streamed`` and clears it after, and the backend reads it while a
turn is in flight.

Safe under concurrency: ``asyncio`` copies the current context when it spawns
the run-loop task, so each agent's task sees the value set on its own behalf,
never a sibling's. Any non-claude-code backend simply ignores it.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any


active_run_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "strix_active_run_context", default=None
)


__all__ = ["active_run_context"]
