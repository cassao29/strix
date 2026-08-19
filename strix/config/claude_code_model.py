"""Claude Code CLI backend — POC.

Routes ``STRIX_LLM=claude-code/<alias>`` through the locally authenticated
``claude`` CLI (``claude /login``, OAuth-backed Claude Pro/Max subscription)
instead of a metered ``ANTHROPIC_API_KEY``. Modeled after
``_CodexResponsesModel`` in :mod:`strix.config.models`, which does the same
thing for the ChatGPT subscription — except Codex reimplements the OAuth
token exchange and calls the Responses API directly, while this drives the
``claude`` binary itself via ``claude_agent_sdk`` (there is no documented
public token-exchange flow for Claude Code to replicate safely).

**v0 scope, intentionally minimal**: one text-in/text-out turn. ``tools`` is
accepted but not translated — this proves the plumbing (routing + local
OAuth session + a real completion coming back) before tool-call
round-tripping is built. All of Claude Code's own tools (Read/Write/Edit/
Bash/...) are disabled so a turn can't do anything but answer in text.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from agents.items import ModelResponse
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import Model
from agents.usage import Usage
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    TextBlock,
    query,
)
from openai.types.responses import ResponseOutputMessage, ResponseOutputText


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.items import TResponseInputItem, TResponseStreamEvent
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from agents.tool import Tool
    from openai.types.responses.response_prompt_param import ResponsePromptParam


logger = logging.getLogger(__name__)

PREFIX = "claude-code"

_NATIVE_TOOLS = ["Read", "Write", "Edit", "NotebookEdit", "Bash", "Glob", "Grep", "WebFetch"]


class ClaudeCodeAuthError(RuntimeError):
    """The local ``claude`` CLI has no active login."""


class ClaudeCodeUnavailableError(RuntimeError):
    """The ``claude`` CLI binary could not be found on ``PATH``."""


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks = [
        str(block.get("text", block)) if isinstance(block, dict) else str(block)
        for block in content
    ]
    return "\n".join(c for c in chunks if c)


def _stringify_input(user_input: str | list[TResponseInputItem]) -> str:
    if isinstance(user_input, str):
        return user_input
    parts: list[str] = []
    for item in user_input:
        role = item.get("role") if isinstance(item, dict) else None
        content = item.get("content") if isinstance(item, dict) else None
        text = _stringify_content(content)
        if text:
            parts.append(f"{role}: {text}" if role else text)
    return "\n\n".join(parts)


class ClaudeCodeModel(Model):
    """One text turn per call, answered by the local Claude Code subscription session."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,  # noqa: ARG002
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,  # noqa: ARG002
        handoffs: list[Handoff],  # noqa: ARG002
        tracing: ModelTracing,  # noqa: ARG002
        *,
        previous_response_id: str | None,  # noqa: ARG002
        conversation_id: str | None,  # noqa: ARG002
        prompt: ResponsePromptParam | None,  # noqa: ARG002
    ) -> ModelResponse:
        # v0: not yet used (no tool-call round-tripping, no prior-response
        # threading, no cross-run conversation state) — required by the Model
        # ABC signature regardless.
        if tools:
            logger.warning(
                "claude-code backend v0: ignoring %d tool(s) — text-only turn", len(tools)
            )

        options = ClaudeAgentOptions(
            model=self.model or None,
            permission_mode="dontAsk",  # deny anything not allow-listed, never prompt
            allowed_tools=[],
            disallowed_tools=list(_NATIVE_TOOLS),
            system_prompt=(
                {"type": "preset", "preset": "claude_code", "append": system_instructions}
                if system_instructions
                else {"type": "preset", "preset": "claude_code"}
            ),
            max_turns=1,
        )

        text_parts: list[str] = []
        usage = Usage()
        turn_error: str | None = None

        try:
            async for message in query(prompt=_stringify_input(input), options=options):
                if isinstance(message, AssistantMessage):
                    text_parts.extend(
                        block.text for block in message.content if isinstance(block, TextBlock)
                    )
                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        turn_error = message.result or message.subtype
                    raw_usage = message.usage or {}
                    in_tok = raw_usage.get("input_tokens", 0) or 0
                    out_tok = raw_usage.get("output_tokens", 0) or 0
                    usage = Usage(
                        input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok
                    )
        except CLINotFoundError as exc:
            raise ClaudeCodeUnavailableError(
                "claude CLI not found on PATH — install Claude Code and run `claude /login`"
            ) from exc
        except ProcessError as exc:
            stderr = (exc.stderr or "").strip()
            if "auth" in stderr.lower() or "login" in stderr.lower():
                raise ClaudeCodeAuthError(
                    f"claude CLI is not authenticated: {stderr or 'run `claude /login`'}"
                ) from exc
            raise

        if turn_error:
            raise RuntimeError(f"claude-code backend turn failed: {turn_error}")

        text = "\n".join(p for p in text_parts if p)
        output_item = ResponseOutputMessage(
            id=f"msg_{uuid.uuid4().hex}",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )
        return ModelResponse(output=[output_item], usage=usage, response_id=FAKE_RESPONSES_ID)

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        # models.py imports this module, so this stays a deferred import to
        # avoid a circular import at module load time.
        from strix.config.models import _completed_stream_event  # noqa: PLC0415

        response = await self.get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
        yield _completed_stream_event(response, self.model)


__all__ = ["PREFIX", "ClaudeCodeAuthError", "ClaudeCodeModel", "ClaudeCodeUnavailableError"]
