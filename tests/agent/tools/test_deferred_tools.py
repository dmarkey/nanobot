"""Tests for deferred tool loading via resolve_tools."""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.resolve import ResolveToolsTool


class _StubTool(Tool):
    def __init__(self, name: str, desc: str = "stub"):
        self._name = name
        self._desc = desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._desc

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class TestToolRegistryDeferred:
    def test_register_deferred_not_in_definitions(self):
        reg = ToolRegistry()
        reg.register(_StubTool("active"), deferred=False)
        reg.register(_StubTool("deferred_one"), deferred=True)
        defs = reg.get_definitions()
        names = [d["function"]["name"] for d in defs]
        assert "active" in names
        assert "deferred_one" not in names

    def test_deferred_catalog(self):
        reg = ToolRegistry()
        reg.register(_StubTool("mcp_a", "Tool A"), deferred=True)
        reg.register(_StubTool("mcp_b", "Tool B"), deferred=True)
        catalog = reg.get_deferred_catalog()
        assert len(catalog) == 2
        assert catalog[0]["name"] == "mcp_a"
        assert catalog[1]["name"] == "mcp_b"

    def test_resolve_promotes_to_active(self):
        reg = ToolRegistry()
        reg.register(_StubTool("mcp_x"), deferred=True)
        assert reg.deferred_count == 1
        resolved = reg.resolve(["mcp_x"])
        assert len(resolved) == 1
        assert resolved[0]["function"]["name"] == "mcp_x"
        assert reg.deferred_count == 0
        names = [d["function"]["name"] for d in reg.get_definitions()]
        assert "mcp_x" in names

    def test_resolve_multiple(self):
        reg = ToolRegistry()
        reg.register(_StubTool("a"), deferred=True)
        reg.register(_StubTool("b"), deferred=True)
        reg.register(_StubTool("c"), deferred=True)
        resolved = reg.resolve(["a", "c"])
        assert len(resolved) == 2
        assert reg.deferred_count == 1

    def test_resolve_nonexistent_returns_empty(self):
        reg = ToolRegistry()
        reg.register(_StubTool("real"), deferred=True)
        resolved = reg.resolve(["fake"])
        assert resolved == []
        assert reg.deferred_count == 1

    def test_has_checks_both(self):
        reg = ToolRegistry()
        reg.register(_StubTool("active"))
        reg.register(_StubTool("lazy"), deferred=True)
        assert reg.has("active")
        assert reg.has("lazy")
        assert not reg.has("missing")

    def test_get_only_returns_active(self):
        reg = ToolRegistry()
        reg.register(_StubTool("lazy"), deferred=True)
        assert reg.get("lazy") is None
        reg.resolve(["lazy"])
        assert reg.get("lazy") is not None

    def test_unregister_removes_deferred(self):
        reg = ToolRegistry()
        reg.register(_StubTool("d"), deferred=True)
        assert reg.deferred_count == 1
        reg.unregister("d")
        assert reg.deferred_count == 0

    def test_disabled_filter_removes_deferred(self):
        reg = ToolRegistry()
        reg.register(_StubTool("mcp_foo"))
        reg.register(_StubTool("mcp_bar"), deferred=True)
        reg.apply_disabled_filter(["mcp_*"])
        assert len(reg.get_definitions()) == 0
        assert reg.deferred_count == 0

    def test_inclusion_filter_keeps_matching_deferred(self):
        reg = ToolRegistry()
        reg.register(_StubTool("keep"), deferred=True)
        reg.register(_StubTool("drop"), deferred=True)
        reg.apply_inclusion_filter(["keep"])
        assert reg.deferred_count == 1
        catalog = reg.get_deferred_catalog()
        assert catalog[0]["name"] == "keep"


class TestResolveToolsTool:
    @pytest.mark.asyncio
    async def test_resolve_returns_schemas(self):
        reg = ToolRegistry()
        reg.register(_StubTool("mcp_a", "A tool"), deferred=True)
        reg.register(_StubTool("mcp_b", "B tool"), deferred=True)
        tool = ResolveToolsTool(reg)
        result = await tool.execute(names=["mcp_a", "mcp_b"])
        assert "Resolved 2 tool(s)" in result
        assert "mcp_a" in result
        assert "mcp_b" in result

    @pytest.mark.asyncio
    async def test_resolve_no_match_shows_available(self):
        reg = ToolRegistry()
        reg.register(_StubTool("mcp_real", "A real tool"), deferred=True)
        tool = ResolveToolsTool(reg)
        result = await tool.execute(names=["mcp_fake"])
        assert "No matching" in result
        assert "mcp_real" in result

    @pytest.mark.asyncio
    async def test_resolve_empty_catalog(self):
        reg = ToolRegistry()
        tool = ResolveToolsTool(reg)
        result = await tool.execute(names=["anything"])
        assert "No deferred tools available" in result

    @pytest.mark.asyncio
    async def test_resolve_shows_remaining_count(self):
        reg = ToolRegistry()
        reg.register(_StubTool("a"), deferred=True)
        reg.register(_StubTool("b"), deferred=True)
        reg.register(_StubTool("c"), deferred=True)
        tool = ResolveToolsTool(reg)
        result = await tool.execute(names=["a"])
        assert "2 deferred tools remaining" in result

    def test_tool_schema(self):
        reg = ToolRegistry()
        tool = ResolveToolsTool(reg)
        schema = tool.to_schema()
        assert schema["function"]["name"] == "resolve_tools"
        assert "names" in schema["function"]["parameters"]["properties"]
