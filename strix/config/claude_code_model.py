"""Claude Code CLI backend.

Routes ``STRIX_LLM=claude-code/<alias>`` through the locally authenticated
``claude`` CLI (``claude /login``, OAuth-backed Claude Pro/Max subscription)
instead of a metered ``ANTHROPIC_API_KEY``. Modeled after
``_CodexResponsesModel`` in :mod:`strix.config.models`, which does the same
thing for the ChatGPT subscription — except Codex reimplements the OAuth
token exchange and calls the Responses API directly, while this drives the
``claude`` binary itself via ``claude_agent_sdk`` (there is no documented
public token-exchange flow for Claude Code to replicate safely).

**Tool-call round-tripping**: the ``tools`` Strix's Runner passes in (its
own pentest tools — shell, proxy, reporting, ...) are re-exposed to Claude
Code as an in-process MCP server (``claude_agent_sdk.create_sdk_mcp_server``)
for the duration of the call, and Claude Code's own native tools
(Read/Write/Edit/Bash/...) stay disabled. This means one Strix "turn" now
maps to Claude Code *autonomously completing the delegated task* — possibly
calling several Strix tools in sequence on its own — before handing back
one final text message, rather than Strix's own Runner executing tool calls
one at a time between turns. That is a deliberate, unavoidable shape
mismatch: a ``Model`` only gets one shot at ``tools``/``get_response`` per
Strix turn, but the ``claude`` CLI's own agentic loop wants to own execution
end to end, so delegation happens at the whole-turn granularity instead of
per-tool-call.

**Run context**: bridged tools run with a real ``ToolContext`` rebuilt from
the run-context dict ``execution.py`` publishes via
:mod:`strix.config.run_context` (coordinator, ``agent_id``, budgets, ...), so
coordination tools (``create_agent``, ``send_message_to_agent``, ...) operate
on the live agent graph. Sandbox shell/filesystem tools bind to their
``SandboxSession`` by closure at construction, not through the context dict,
so they execute in the real sandbox regardless. Outside a run (e.g. a bare
unit test) the context falls back to an empty dict.

**Tool coverage**: both ``FunctionTool`` and ``CustomTool`` are bridged —
``CustomTool`` (e.g. ``apply_patch``) is fronted with Strix's own
``_custom_tool_as_function_tool`` adapter so its single raw-string payload
maps onto an MCP JSON-schema tool. A wall-clock timeout
(``_TURN_TIMEOUT_S``) bounds a wedged turn.

**Finalization handshake**: lifecycle tools (``finish_scan`` / ``agent_finish``)
are terminal in Strix — its Runner stops an agent the moment one succeeds. Since
Claude Code runs its own loop, the bridge intercepts a lifecycle call (records
args, returns a placeholder, runs nothing, stops consuming the CLI turn) and the
turn re-emits it as a real function call. Strix's Runner then executes it exactly
once, its ``tool_use_behavior`` produces the terminal ``final_output``, and the
run ends cleanly — no work past the finish and no spurious "ended without calling
finish_scan" error.

**Residual mismatch worth knowing**: because the ``claude`` CLI owns its own
agentic loop, one Strix turn is a whole delegated task rather than a single
model step, so Strix's own per-turn budget/compaction/turn-guard machinery
sees one long turn instead of many, and token usage under-counts (only the
final ``ResultMessage`` per turn is captured). Cost reads as unmetered ($0): a
subscription genuinely has no per-token price, and ``claude-code/*`` is not in
LiteLLM's price table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from agents.items import ModelResponse
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import Model
from agents.tool import CustomTool, FunctionTool
from agents.tool_context import ToolContext
from agents.usage import Usage
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
)
from claude_agent_sdk import tool as sdk_tool
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from strix.config.run_context import active_run_context


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.items import TResponseInputItem, TResponseStreamEvent
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from agents.tool import Tool
    from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool
    from openai.types.responses.response_prompt_param import ResponsePromptParam


logger = logging.getLogger(__name__)

PREFIX = "claude-code"

_NATIVE_TOOLS = ["Read", "Write", "Edit", "NotebookEdit", "Bash", "Glob", "Grep", "WebFetch"]

_MCP_SERVER_NAME = "strix"
# Bound so a misbehaving delegated task can't run forever; generous because
# one Strix turn now covers a whole delegated task, not a single tool call.
_MAX_TURNS_WITH_TOOLS = 40
# Lifecycle tools whose call ends a Strix agent. When Claude Code calls one over
# MCP the bridge intercepts it (records args, does NOT run it) and the turn
# re-emits it as a real tool call so Strix's Runner executes it once and its
# ``tool_use_behavior`` stops the run cleanly — see the module docstring.
_LIFECYCLE_TOOLS = frozenset({"finish_scan", "agent_finish"})
# Wall-clock ceiling for a single delegated turn. A whole task can chain many
# tool calls, so this is generous — it exists to stop a wedged CLI process
# hanging the scan forever, not to bound normal work.
_TURN_TIMEOUT_S = 1800.0


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


def _mcp_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


class _LifecycleCapture:
    """Records the first lifecycle tool call intercepted during one turn.

    Lifecycle tools (``finish_scan`` / ``agent_finish``) are terminal: Strix's
    Runner stops the agent the moment one succeeds. If Claude Code executed one
    itself over MCP, Strix's Runner would never see the tool call and would (a)
    log a spurious "ended without calling finish_scan" error and (b) let Claude
    Code keep working past the finish. So the bridge intercepts the call —
    records its args, returns a success placeholder, runs nothing — and the
    turn re-emits it as a real function call for Strix to execute once.
    """

    def __init__(self) -> None:
        self.name: str | None = None
        self.arguments: str | None = None

    @property
    def captured(self) -> bool:
        return self.name is not None

    def record(self, name: str, arguments: str) -> None:
        if self.name is None:  # keep the first; ignore any trailing extras
            self.name = name
            self.arguments = arguments


def _make_bridge_tool(tool: FunctionTool, capture: _LifecycleCapture) -> SdkMcpTool[Any]:
    """Wrap one Strix ``FunctionTool`` as an in-process MCP tool.

    Non-lifecycle tools invoke the real ``on_invoke_tool`` — the same coroutine
    Strix's own Runner would call — so they actually run (shell commands execute
    in the sandbox, proxy requests go out, reports get filed). The ``ToolContext``
    is rebuilt from the run context published by ``execution.py``
    (:mod:`strix.config.run_context`), so coordination tools that read
    ``context["coordinator"]`` see the real graph; outside a run it falls back to
    an empty dict. Lifecycle tools are intercepted into ``capture`` instead of run
    (see :class:`_LifecycleCapture`).
    """
    is_lifecycle = tool.name in _LIFECYCLE_TOOLS

    async def _invoke(args: dict[str, Any]) -> dict[str, Any]:
        args_json = json.dumps(args, ensure_ascii=False)
        if is_lifecycle:
            capture.record(tool.name, args_json)
            return _mcp_result(
                json.dumps({"success": True, "message": "Recorded; Strix is finalizing the run."})
            )
        call_id = f"claude_code_{uuid.uuid4().hex}"
        run_context = active_run_context.get() or {}
        ctx: ToolContext[Any] = ToolContext(
            context=run_context,
            tool_name=tool.name,
            tool_call_id=call_id,
            tool_arguments=args_json,
        )
        try:
            result = await tool.on_invoke_tool(ctx, args_json)
        except Exception as exc:  # noqa: BLE001 - report to Claude, don't kill the turn
            logger.warning("claude-code bridge: tool %r raised", tool.name, exc_info=True)
            return _mcp_result(f"Tool {tool.name!r} failed: {exc}", is_error=True)
        return _mcp_result(result if isinstance(result, str) else json.dumps(result, default=str))

    return sdk_tool(tool.name, tool.description or tool.name, tool.params_json_schema)(_invoke)


def _collect_assistant_blocks(message: AssistantMessage, text_parts: list[str]) -> int:
    """Append text blocks to ``text_parts`` in place; return the tool-call count."""
    tool_calls = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            tool_calls += 1
            logger.debug("claude-code turn: tool call %s(%r)", block.name, block.input)
        elif isinstance(block, ToolResultBlock) and block.is_error:
            logger.debug("claude-code turn: tool error on %s", block.tool_use_id)
    return tool_calls


def _usage_from_result(message: ResultMessage) -> tuple[Usage, str | None]:
    raw_usage = message.usage or {}
    in_tok = raw_usage.get("input_tokens", 0) or 0
    out_tok = raw_usage.get("output_tokens", 0) or 0
    usage = Usage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok)
    error = (message.result or message.subtype) if message.is_error else None
    return usage, error


def _as_function_tool(tool: Tool) -> FunctionTool | None:
    """Coerce a Strix tool into a bridgeable ``FunctionTool``, or ``None`` to skip.

    ``CustomTool`` (e.g. ``apply_patch``) takes a single raw-string payload
    rather than JSON args; Strix already ships an adapter that fronts it with a
    ``{field: str}`` JSON schema for the chat-completions path, so reuse that
    exact conversion instead of reinventing it.
    """
    if isinstance(tool, FunctionTool):
        return tool
    if isinstance(tool, CustomTool):
        from strix.agents.factory import _custom_tool_as_function_tool  # noqa: PLC0415

        return _custom_tool_as_function_tool(tool)
    logger.warning("claude-code bridge: skipping unsupported tool type %s", type(tool).__name__)
    return None


def _build_mcp_bridge(tools: list[Tool], capture: _LifecycleCapture) -> McpSdkServerConfig | None:
    bridged = [ft for ft in (_as_function_tool(t) for t in tools) if ft is not None]
    if not bridged:
        return None
    return create_sdk_mcp_server(
        name=_MCP_SERVER_NAME, tools=[_make_bridge_tool(t, capture) for t in bridged]
    )


class ClaudeCodeModel(Model):
    """One Strix turn, answered by the local Claude Code subscription session.

    When ``tools`` is non-empty, Claude Code gets them as an in-process MCP
    server and may call several before answering (see module docstring).
    """

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
        # Not yet used (no prior-response threading, no cross-run conversation
        # state) — required by the Model ABC signature regardless.
        capture = _LifecycleCapture()
        mcp_server = _build_mcp_bridge(tools, capture)

        options = ClaudeAgentOptions(
            model=self.model or None,
            permission_mode="dontAsk",  # deny anything not allow-listed, never prompt
            allowed_tools=[f"mcp__{_MCP_SERVER_NAME}__*"] if mcp_server else [],
            disallowed_tools=list(_NATIVE_TOOLS),
            mcp_servers={_MCP_SERVER_NAME: mcp_server} if mcp_server else {},
            system_prompt=(
                {"type": "preset", "preset": "claude_code", "append": system_instructions}
                if system_instructions
                else {"type": "preset", "preset": "claude_code"}
            ),
            max_turns=_MAX_TURNS_WITH_TOOLS if mcp_server else 1,
        )

        text_parts: list[str] = []
        usage = Usage()
        turn_error: str | None = None
        tool_calls = 0

        try:
            async with asyncio.timeout(_TURN_TIMEOUT_S):
                async for message in query(prompt=_stringify_input(input), options=options):
                    if isinstance(message, AssistantMessage):
                        tool_calls += _collect_assistant_blocks(message, text_parts)
                    elif isinstance(message, ResultMessage):
                        usage, turn_error = _usage_from_result(message)
                    # A lifecycle tool is terminal: stop consuming the CLI turn so
                    # Claude Code can't keep working past the finish. Exiting the
                    # loop closes the query generator, ending the CLI turn.
                    if capture.captured:
                        break
        except TimeoutError as exc:
            raise RuntimeError(
                f"claude-code backend turn exceeded {_TURN_TIMEOUT_S:.0f}s and was aborted"
            ) from exc
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

        if tool_calls:
            logger.info("claude-code turn: completed after %d tool call(s)", tool_calls)

        # A captured lifecycle call is re-emitted as a real function call so
        # Strix's Runner executes it once and its tool_use_behavior stops the run
        # authoritatively — no spurious "ended without finish_scan" and no work
        # past the finish.
        if capture.captured:
            logger.info("claude-code turn: finalizing via %s", capture.name)
            finish_call = ResponseFunctionToolCall(
                id=f"fc_{uuid.uuid4().hex}",
                call_id=f"call_{uuid.uuid4().hex}",
                name=capture.name or "",
                arguments=capture.arguments or "{}",
                type="function_call",
            )
            return ModelResponse(output=[finish_call], usage=usage, response_id=FAKE_RESPONSES_ID)

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
