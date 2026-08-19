#!/usr/bin/env python3
"""POC smoke test for tool-call round-tripping in the claude-code backend.

Registers one trivial FunctionTool (add two numbers) using the real SDK
`function_tool` decorator, passes it into ClaudeCodeModel.get_response() as
Strix's Runner would, and checks the local `claude` CLI actually calls it
through the in-process MCP bridge (not just answers from its own reasoning).

Usage: uv run python scripts/poc_claude_code_tool_roundtrip.py
"""

import asyncio
import logging

from agents import function_tool
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

from strix.config.claude_code_model import ClaudeCodeModel


logger = logging.getLogger("poc_claude_code_tool_roundtrip")

_CALLS: list[tuple[int, int]] = []


@function_tool
def add_numbers(a: int, b: int) -> str:
    """Add two integers and return the sum as a string."""
    _CALLS.append((a, b))
    return str(a + b)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = ClaudeCodeModel("sonnet")
    response = await model.get_response(
        system_instructions=(
            "You must use the add_numbers tool for any arithmetic instead of "
            "computing it yourself. Reply with only the final numeric result."
        ),
        input="What is 482913 plus 117359? Use the tool.",
        model_settings=ModelSettings(),
        tools=[add_numbers],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    text = response.output[0].content[0].text
    logger.info("response text: %r", text)
    logger.info("tool actually invoked with: %r", _CALLS)
    logger.info("usage: %s", response.usage)

    expected = 482913 + 117359
    if _CALLS and str(expected) in text:
        logger.info("PASS: tool was called and result reflects its real output")
    else:
        logger.info("FAIL: tool round-trip did not happen as expected")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
