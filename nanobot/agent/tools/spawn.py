"""Spawn tool for creating background subagents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.workspace_access import current_workspace_scope

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


@tool_parameters(
    tool_parameters_schema(
        task=StringSchema("The task for the subagent to complete"),
        label=StringSchema("Optional short label for the task (for display)"),
        tools=ArraySchema(
            StringSchema("Tool name or glob pattern (e.g. 'mcp_nta_*')"),
            description=(
                "Optional list of tool names or glob patterns to enable for this subagent. "
                "Only matching tools will be available. Omit to give the subagent all tools."
            ),
        ),
        temperature=NumberSchema(
            description=(
                "Optional sampling temperature for the subagent "
                "(0.0 = deterministic, higher = more creative). "
                "Defaults to the provider's configured temperature."
            ),
            minimum=0.0,
            maximum=2.0,
        ),
        detached=BooleanSchema(
            description=(
                "Set true for long-running, fire-and-forget work (e.g. a download). "
                "A detached subagent does NOT hold the current turn open; its result "
                "is delivered as a new message when it finishes, so you stay responsive. "
                "Omit/false for quick helpers whose result you want back in this turn."
            ),
        ),
        required=["task"],
    )
)
class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(manager=ctx.subagent_manager)

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful. "
            "Use the 'tools' parameter to restrict which tools the subagent can use. "
            "For long-running work (e.g. downloads) set 'detached' true so you stay "
            "responsive and get the result as a new message when it finishes."
        )

    async def execute(
        self,
        task: str,
        label: str | None = None,
        tools: list[str] | None = None,
        temperature: float | None = None,
        detached: bool = False,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        running = self._manager.get_running_count()
        limit = self._manager.max_concurrent_subagents
        if running >= limit:
            return (
                f"Cannot spawn subagent: concurrency limit reached "
                f"({running}/{limit} running). Wait for a running subagent "
                f"to complete before spawning a new one."
            )
        request_ctx = current_request_context()
        if request_ctx is None or request_ctx.runtime is None:
            return ToolResult.error("Error: spawn requires an active model runtime")
        origin_channel = request_ctx.channel
        origin_chat_id = request_ctx.chat_id
        session_key = request_ctx.session_key or f"{origin_channel}:{origin_chat_id}"
        return await self._manager.spawn(
            task=task,
            runtime=request_ctx.runtime,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=session_key,
            tool_filter=tools,
            origin_message_id=request_ctx.message_id,
            temperature=temperature,
            detached=detached,
            workspace_scope=current_workspace_scope(),
        )
