"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import ArraySchema, StringSchema, tool_parameters_schema

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
                "Only matching tools will be available. Omit to give the subagent all tools. "
                "Builtin tools (read_file, exec, web_search, etc.) are always available "
                "unless disabled by config — this filter applies to MCP tools."
            ),
        ),
        required=["task"],
    )
)
class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

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
            "Use the 'tools' parameter to restrict which MCP tools the subagent "
            "can use (e.g. ['mcp_nta_*'] for transport tools only)."
        )

    async def execute(self, task: str, label: str | None = None, tools: list[str] | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
            tool_filter=tools,
        )
