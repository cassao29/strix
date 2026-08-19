"""Unit tests for the claude-code backend (``strix.config.claude_code_model``).

These mock ``claude_agent_sdk.query`` so they need neither the ``claude`` CLI
nor a network/subscription, and they exercise the pieces that carry real risk:
the MCP tool bridge (FunctionTool + CustomTool), run-context threading, and
error/timeout mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from agents import RunContextWrapper, function_tool
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.tool import CustomTool, FunctionTool
from claude_agent_sdk import AssistantMessage, ProcessError, ResultMessage, TextBlock

import strix.config.claude_code_model as ccm
from strix.config.claude_code_model import (
    ClaudeCodeAuthError,
    ClaudeCodeModel,
    _as_function_tool,
    _build_mcp_bridge,
    _make_bridge_tool,
)
from strix.config.run_context import active_run_context


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="sonnet")


def _result(*, is_error: bool = False, result: str | None = None) -> ResultMessage:
    return ResultMessage(
        subtype="success" if not is_error else "error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="s1",
        result=result,
        usage={"input_tokens": 3, "output_tokens": 4},
    )


async def _invoke_bridge(bridge_tool: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Call the underlying async handler of an SdkMcpTool."""
    return await bridge_tool.handler(args)


async def test_bridge_tool_runs_real_callable() -> None:
    # Assert our bridge invokes the real coroutine behind a FunctionTool.
    @function_tool
    def echo_ctx(value: str) -> str:
        return f"echo:{value}"

    bridge = _make_bridge_tool(echo_ctx)
    out = await _invoke_bridge(bridge, {"value": "hi"})

    assert out["is_error"] is False
    assert "echo:hi" in out["content"][0]["text"]


async def test_bridge_tool_threads_context_into_tool() -> None:
    captured: dict[str, Any] = {}

    @function_tool
    def grab(ctx: RunContextWrapper, value: str) -> str:  # noqa: ARG001 - value in schema
        captured["ctx"] = ctx.context
        return "ok"

    bridge = _make_bridge_tool(grab)
    token = active_run_context.set({"coordinator": "COORD", "agent_id": "a1"})
    try:
        await _invoke_bridge(bridge, {"value": "x"})
    finally:
        active_run_context.reset(token)

    assert captured["ctx"] == {"coordinator": "COORD", "agent_id": "a1"}


async def test_bridge_tool_reports_exception_as_error_result() -> None:
    # A raw FunctionTool whose invoker actually raises (function_tool would
    # otherwise swallow the error into a string itself) — proves the bridge
    # turns a genuine exception into an MCP error result instead of crashing.
    async def _raise(_ctx: Any, _args: str) -> str:
        raise ValueError("kaboom")

    raw = FunctionTool(
        name="boom",
        description="always fails",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke_tool=_raise,
    )
    bridge = _make_bridge_tool(raw)
    out = await _invoke_bridge(bridge, {})
    assert out["is_error"] is True
    assert "kaboom" in out["content"][0]["text"]


def test_custom_tool_is_bridged_not_skipped() -> None:
    async def _run(_ctx: Any, raw: str) -> str:
        return f"patched:{raw}"

    custom = CustomTool(name="apply_patch", description="apply a patch", on_invoke_tool=_run)
    bridged = _as_function_tool(custom)
    assert bridged is not None
    assert bridged.name == "apply_patch"
    # Strix's adapter fronts the raw-string payload with a JSON-schema field.
    assert bridged.params_json_schema["type"] == "object"

    server = _build_mcp_bridge([custom])
    assert server is not None  # not dropped


def test_build_mcp_bridge_empty_when_no_tools() -> None:
    assert _build_mcp_bridge([]) is None


async def test_text_only_turn_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        captured["prompt"] = prompt
        captured["options"] = options
        yield _assistant("PONG")
        yield _result()

    monkeypatch.setattr(ccm, "query", fake_query)
    model = ClaudeCodeModel("sonnet")
    resp = await _get(model, tools=[])

    assert resp.output[0].content[0].text == "PONG"
    assert resp.usage.input_tokens == 3
    assert resp.usage.output_tokens == 4
    # No tools → single-shot turn.
    assert captured["options"].max_turns == 1
    assert captured["options"].mcp_servers == {}


async def test_turn_with_tools_enables_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:  # noqa: ARG001
        captured["options"] = options
        yield _assistant("done")
        yield _result()

    @function_tool
    def noop(value: str) -> str:  # noqa: ARG001 - value required by schema
        return "ok"

    monkeypatch.setattr(ccm, "query", fake_query)
    model = ClaudeCodeModel("sonnet")
    await _get(model, tools=[noop])

    opts = captured["options"]
    assert opts.max_turns > 1
    assert "strix" in opts.mcp_servers
    assert any("mcp__strix__" in t for t in opts.allowed_tools)


async def test_auth_error_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:  # noqa: ARG001
        raise ProcessError("failed", exit_code=1, stderr="please run /login first")
        yield  # unreachable; make this an async generator

    monkeypatch.setattr(ccm, "query", fake_query)
    model = ClaudeCodeModel("sonnet")
    with pytest.raises(ClaudeCodeAuthError):
        await _get(model, tools=[])


async def test_turn_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:  # noqa: ARG001
        yield _result(is_error=True, result="boom")

    monkeypatch.setattr(ccm, "query", fake_query)
    model = ClaudeCodeModel("sonnet")
    with pytest.raises(RuntimeError, match="turn failed"):
        await _get(model, tools=[])


async def _get(model: ClaudeCodeModel, *, tools: list[Any]) -> Any:
    return await model.get_response(
        system_instructions=None,
        input="go",
        model_settings=ModelSettings(),
        tools=tools,
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
