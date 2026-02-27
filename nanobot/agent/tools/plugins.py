"""Plugin loader for auto-discovering custom tools from workspace/tools/."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolContext:
    """Context passed to plugin tool constructors."""

    workspace: Path
    allowed_dir: Path | None
    working_dir: str
    exec_timeout: int
    restrict_to_workspace: bool
    brave_api_key: str | None


class PluginLoader:
    """Discovers and loads Tool subclasses from ``{workspace}/tools/*.py``."""

    def __init__(self, workspace: Path, context: ToolContext) -> None:
        self.tools_dir = workspace / "tools"
        self.context = context
        self._cache: dict[str, Tool] | None = None

    def load(self) -> dict[str, Tool]:
        """Scan the tools directory and return a name->Tool mapping.

        Results are cached after the first call. Files prefixed with ``_``
        are skipped. Errors in individual plugins are logged and skipped.
        """
        if self._cache is not None:
            return self._cache

        tools: dict[str, Tool] = {}

        if not self.tools_dir.is_dir():
            self._cache = tools
            return tools

        # Ensure tools dir is on sys.path so plugins can import helpers
        # (e.g. _base.py) from the same directory.
        tools_str = str(self.tools_dir)
        if tools_str not in sys.path:
            sys.path.insert(0, tools_str)

        for path in sorted(self.tools_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                self._load_file(path, tools)
            except Exception as e:
                logger.warning("Failed to load plugin {}: {}", path.name, e)

        self._cache = tools
        return tools

    def register_into(self, registry: ToolRegistry) -> None:
        """Load plugins and register into *registry*, skipping existing names."""
        for name, tool in self.load().items():
            if registry.has(name):
                logger.debug("Plugin tool '{}' skipped -- built-in takes priority", name)
                continue
            registry.register(tool)
            logger.info("Registered plugin tool '{}'", name)

    # ------------------------------------------------------------------

    def _load_file(self, path: Path, tools: dict[str, Tool]) -> None:
        """Import a single .py file and collect Tool subclasses from it."""
        module_name = f"nanobot_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Cannot create module spec for {}", path.name)
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, Tool) or obj is Tool:
                continue
            if inspect.isabstract(obj):
                continue
            try:
                instance = self._instantiate(obj)
            except Exception as e:
                logger.warning("Failed to instantiate {} from {}: {}", obj.__name__, path.name, e)
                continue

            name = instance.name
            if name in tools:
                logger.debug("Duplicate plugin tool name '{}' in {} -- skipped", name, path.name)
                continue
            tools[name] = instance

    def _instantiate(self, cls: type[Tool]) -> Tool:
        """Try to construct a Tool, first with context= kwarg, then no-arg."""
        try:
            return cls(context=self.context)
        except TypeError:
            return cls()
