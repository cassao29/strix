#!/usr/bin/env python3
"""POC: bridge a REAL Strix tool (not a synthetic one) through the claude-code backend.

Uses strix.tools.notes.tools.create_note / list_notes — real production
FunctionTools with Strix's actual Pydantic-flavored JSON schema and
RunContextWrapper-reading implementation — to confirm the MCP bridge in
strix/config/claude_code_model.py handles Strix's real tool shapes, not just
a hand-rolled example.

Usage: uv run python scripts/poc_claude_code_real_strix_tool.py
"""

import asyncio
import logging

from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

from strix.config.claude_code_model import ClaudeCodeModel
from strix.tools.notes.tools import _notes_storage, create_note, list_notes


logger = logging.getLogger("poc_claude_code_real_strix_tool")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = ClaudeCodeModel("sonnet")
    marker = "strix-mcp-bridge-poc"
    response = await model.get_response(
        system_instructions=(
            f"Call create_note to save one note in the 'general' category whose "
            f"content is exactly '{marker}', then call list_notes and report how "
            f"many notes exist."
        ),
        input="Do it now.",
        model_settings=ModelSettings(),
        tools=[create_note, list_notes],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    text = response.output[0].content[0].text
    logger.info("response text: %r", text)
    logger.info("real _notes_storage after the turn: %s", _notes_storage)

    saved = any(n.get("content") == marker for n in _notes_storage.values())
    logger.info("PASS: real Strix tool wrote through the bridge" if saved else "FAIL")
    if not saved:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
