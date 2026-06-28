"""Tool registry for dynamic tool management."""

import json
from fnmatch import fnmatch
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._deferred: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool, *, deferred: bool = False) -> None:
        """Register a tool. Deferred tools are held back until resolved."""
        if deferred:
            self._deferred[tool.name] = tool
        else:
            self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._deferred.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        """Get a tool by name (active tools only)."""
        return self._tools.get(name)

    @staticmethod
    def _lookup_key(name: str) -> str:
        """Normalize names for suggestions only; never for execution."""
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _suggest_name(self, name: str) -> str | None:
        key = self._lookup_key(str(name or ""))
        if not key:
            return None
        matches = [
            registered
            for registered in self._tools
            if self._lookup_key(registered) == key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def has(self, name: str) -> bool:
        """Check if a tool is registered (active or deferred)."""
        return name in self._tools or name in self._deferred

    def resolve(self, names: list[str]) -> list[dict[str, Any]]:
        """Promote deferred tools to active. Returns their full schemas."""
        resolved: list[dict[str, Any]] = []
        for name in names:
            if name in self._deferred:
                tool = self._deferred.pop(name)
                self._tools[name] = tool
                resolved.append(tool.to_schema())
                self._cached_definitions = None
                logger.info("Resolved deferred tool '{}'", name)
        return resolved

    def get_deferred_catalog(self) -> list[dict[str, str]]:
        """Name + description pairs for deferred tools (for system prompt injection)."""
        return [
            {"name": t.name, "description": t.description}
            for t in sorted(self._deferred.values(), key=lambda t: t.name)
        ]

    @property
    def deferred_count(self) -> int:
        return len(self._deferred)

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """Extract a normalized tool name from either OpenAI or flat schemas."""
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions with stable ordering for cache-friendly prompts.

        Built-in tools are sorted first as a stable prefix, then MCP tools are
        sorted and appended.  The result is cached until the next
        register/unregister call.
        """
        if self._cached_definitions is not None:
            return self._cached_definitions

        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []
        for schema in definitions:
            name = self._schema_name(schema)
            if name.startswith("mcp_"):
                mcp_tools.append(schema)
            else:
                builtins.append(schema)

        builtins.sort(key=self._schema_name)
        mcp_tools.sort(key=self._schema_name)
        self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    def prepare_call(
        self,
        name: str,
        params: Any,
    ) -> tuple[Tool | None, Any, str | None]:
        """Resolve, cast, and validate one tool call."""
        tool = self._tools.get(name)
        if not tool:
            suggestion = self._suggest_name(str(name))
            hint = f" Did you mean '{suggestion}'? Tool names must match exactly." if suggestion else ""
            return None, params, (
                f"Error: Tool '{name}' not found.{hint} Available: {', '.join(self.tool_names)}"
            )

        params = self._coerce_params(tool, params)
        if not isinstance(params, dict):
            return tool, params, (
                f"Error: Tool '{name}' parameters must be a JSON object, got "
                f"{type(params).__name__}. Use named parameters like "
                'tool_name(param1="value1", param2="value2") matching the tool schema.'
            )

        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, (
                f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            )
        return tool, cast_params, None

    @classmethod
    def _coerce_argument_value(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return {}

        if not stripped.startswith(("{", "[")):
            return value

        try:
            parsed = json.loads(stripped)
        except Exception:
            return value

        return parsed

    @classmethod
    def _coerce_params(cls, tool: Tool, params: Any) -> Any:
        params = cls._coerce_argument_value(params)
        return cls._unwrap_arguments_payload(tool, params)

    @classmethod
    def _unwrap_arguments_payload(cls, tool: Tool, params: Any) -> Any:
        if not isinstance(params, dict) or set(params) != {"arguments"}:
            return params
        properties = (tool.parameters or {}).get("properties", {})
        if isinstance(properties, dict) and "arguments" in properties:
            return params
        return cls._coerce_argument_value(params.get("arguments"))

    async def execute(self, name: str, params: Any) -> Any:
        """Execute a tool by name with given parameters."""
        hint = "\n\n[Analyze the error above and try a different approach.]"
        tool, params, error = self.prepare_call(name, params)
        if error:
            return error + hint

        try:
            assert tool is not None  # guarded by prepare_call()
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + hint
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + hint

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def apply_disabled_filter(self, patterns: list[str]) -> None:
        """Remove tools whose names match any of the given patterns (exact or glob)."""
        if not patterns:
            return
        logger.info("Applying disabled tool filter: patterns={}, registered={}", patterns, list(self._tools.keys()))
        to_remove = [
            name for name in self._tools
            if any(fnmatch(name, p) for p in patterns)
        ]
        for name in to_remove:
            del self._tools[name]
            logger.info("Tool '{}' disabled by filter", name)
        deferred_remove = [
            name for name in self._deferred
            if any(fnmatch(name, p) for p in patterns)
        ]
        for name in deferred_remove:
            del self._deferred[name]
            logger.info("Deferred tool '{}' disabled by filter", name)
        if not to_remove and not deferred_remove:
            logger.warning("Disabled tool filter matched nothing: patterns={}", patterns)

    def apply_inclusion_filter(self, patterns: list[str]) -> None:
        """Keep only tools whose names match at least one pattern (exact or glob). Others are removed."""
        if not patterns:
            return
        to_remove = [
            name for name in self._tools
            if not any(fnmatch(name, p) for p in patterns)
        ]
        for name in to_remove:
            del self._tools[name]
        deferred_remove = [
            name for name in self._deferred
            if not any(fnmatch(name, p) for p in patterns)
        ]
        for name in deferred_remove:
            del self._deferred[name]
        total_removed = len(to_remove) + len(deferred_remove)
        total_kept = len(self._tools) + len(self._deferred)
        logger.info("Inclusion filter kept {} tools, removed {}: patterns={}", total_kept, total_removed, patterns)
