#!/usr/bin/env python3
"""POC smoke test for the claude-code backend (strix/config/claude_code_model.py).

Calls ClaudeCodeModel.get_response() directly with a trivial prompt,
bypassing Strix's scan orchestrator entirely, to validate: the `claude` CLI
is authenticated (subscription OAuth, no ANTHROPIC_API_KEY needed) and a
real completion round-trips through the Model interface Strix's agent
runtime expects.

Usage: uv run python scripts/poc_claude_code_backend.py
"""

import asyncio
import logging

from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

from strix.config.claude_code_model import (
    ClaudeCodeAuthError,
    ClaudeCodeModel,
    ClaudeCodeUnavailableError,
)


logger = logging.getLogger("poc_claude_code_backend")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = ClaudeCodeModel("sonnet")
    try:
        response = await model.get_response(
            system_instructions="You are a terse test responder.",
            input="Reply with exactly: PONG",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    except ClaudeCodeUnavailableError:
        logger.exception("FAIL: claude CLI not found")
        raise SystemExit(1) from None
    except ClaudeCodeAuthError:
        logger.exception("FAIL: claude CLI not authenticated")
        raise SystemExit(1) from None

    text = response.output[0].content[0].text
    logger.info("response text: %r", text)
    logger.info("usage: %s", response.usage)
    logger.info("PASS" if "PONG" in text else "UNEXPECTED OUTPUT")


if __name__ == "__main__":
    asyncio.run(main())
