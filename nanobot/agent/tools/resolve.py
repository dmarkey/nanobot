"""Meta-tool for resolving deferred tool definitions on demand."""

import json
from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.schema import ArraySchema, StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        names=ArraySchema(
            StringSchema("Tool name from the deferred tools catalog"),
            description="Tool names to resolve. You can resolve multiple tools in one call.",
        ),
        required=["names"],
    )
)
class ResolveToolsTool(Tool):
    """Meta-tool that promotes deferred tools to active so the LLM can call them."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    @property
    def name(self) -> str:
        return "resolve_tools"

    @property
    def description(self) -> str:
        return (
            "Fetch full definitions for deferred tools so they can be called. "
            "Pass one or more tool names from the deferred tools catalog listed in the system prompt. "
            "After resolving, the tools become available for use on your next action."
        )

    async def execute(self, *, names: list[str], **kwargs: Any) -> str:
        resolved = self._registry.resolve(names)
        if not resolved:
            catalog = self._registry.get_deferred_catalog()
            if catalog:
                available = ", ".join(e["name"] for e in catalog)
                return f"No matching deferred tools found for: {', '.join(names)}. Available: {available}"
            return "No deferred tools available. The requested tools may already be active or do not exist."

        lines: list[str] = []
        for schema in resolved:
            fn = schema["function"]
            lines.append(f"## {fn['name']}")
            if fn.get("description"):
                lines.append(fn["description"])
            lines.append(f"```json\n{json.dumps(fn['parameters'], indent=2)}\n```")
            lines.append("")

        remaining = self._registry.deferred_count
        suffix = f"\n({remaining} deferred tools remaining)" if remaining else ""
        return (
            f"Resolved {len(resolved)} tool(s) — now available for calling:\n\n"
            + "\n".join(lines)
            + suffix
        )
