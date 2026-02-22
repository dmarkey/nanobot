"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool

# Mapping from tool name to the class that implements it
_TOOL_NAME_MAP: dict[str, str] = {
    "read_file": "ReadFileTool",
    "write_file": "WriteFileTool",
    "edit_file": "EditFileTool",
    "list_dir": "ListDirTool",
    "exec": "ExecTool",
    "web_search": "WebSearchTool",
    "web_fetch": "WebFetchTool",
}


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        subagent_profiles: dict | None = None,
        default_max_iterations: int = 15,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.subagent_profiles = subagent_profiles or {}
        self.default_max_iterations = default_max_iterations
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        profile: str | None = None,
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.
            profile: Optional subagent profile name.

        Returns:
            Status message indicating the subagent was started.
        """
        if profile and profile not in self.subagent_profiles:
            available = ", ".join(sorted(self.subagent_profiles)) or "(none)"
            return f"Error: Unknown profile '{profile}'. Available profiles: {available}"

        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, profile)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        profile_hint = f", profile: {profile}" if profile else ""
        logger.info("Spawned subagent [{}{}]: {}", task_id, profile_hint, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        profile: str | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            # Resolve profile config
            profile_config = self.subagent_profiles.get(profile) if profile else None

            # Determine model and iteration limit
            model = self.model
            max_iterations = self.default_max_iterations
            if profile_config:
                if profile_config.model:
                    model = profile_config.model
                max_iterations = profile_config.max_iterations

            # Build tools
            if profile_config and profile_config.tools:
                tools = self._register_profile_tools(profile_config.tools)
            else:
                tools = self._register_default_subagent_tools()

            # Load skills content if profile specifies them
            skills_content = ""
            if profile_config and profile_config.skills:
                skills_content = self._load_skills(profile_config.skills)

            # Determine capability flags from registered tools
            has_exec = tools.has("exec")
            has_web = tools.has("web_search") or tools.has("web_fetch")
            has_files = tools.has("read_file") or tools.has("write_file")

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(
                task,
                has_exec=has_exec,
                has_web=has_web,
                has_files=has_files,
                skills_content=skills_content,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations)
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })

                    # Execute tools
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug("Subagent [{}] executing: {} with arguments: {}", task_id, tool_call.name, args_str)
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = (
                    f"Task did NOT complete: the subagent ran out of iterations "
                    f"(max {max_iterations}). The task may need to be broken into "
                    f"smaller pieces, or the iteration limit needs to be increased."
                )
                logger.warning("Subagent [{}] exhausted max iterations ({})", task_id, max_iterations)
                await self._announce_result(task_id, label, task, final_result, origin, "error")
            else:
                logger.info("Subagent [{}] completed successfully", task_id)
                await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    def _register_default_subagent_tools(self) -> ToolRegistry:
        """Register the default tool set for subagents (no message, no spawn)."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enabled:
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ))
        tools.register(WebSearchTool(api_key=self.brave_api_key))
        tools.register(WebFetchTool())
        return tools

    def _register_profile_tools(self, tool_names: list[str]) -> ToolRegistry:
        """Register only the whitelisted tools from a profile."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None

        for name in tool_names:
            if name not in _TOOL_NAME_MAP:
                logger.warning("Unknown tool '{}' in subagent profile, skipping", name)
                continue

            # exec is always gated by the global enabled flag
            if name == "exec" and not self.exec_config.enabled:
                logger.debug("Tool 'exec' requested by profile but globally disabled, skipping")
                continue

            if name == "read_file":
                tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            elif name == "write_file":
                tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            elif name == "edit_file":
                tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            elif name == "list_dir":
                tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
            elif name == "exec":
                tools.register(ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                ))
            elif name == "web_search":
                tools.register(WebSearchTool(api_key=self.brave_api_key))
            elif name == "web_fetch":
                tools.register(WebFetchTool())

        return tools

    def _load_skills(self, skill_names: list[str]) -> str:
        """Load skills content for inclusion in the subagent prompt."""
        from nanobot.agent.skills import SkillsLoader
        loader = SkillsLoader(self.workspace)
        return loader.load_skills_for_context(skill_names)

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    def _build_subagent_prompt(
        self,
        task: str,
        *,
        has_exec: bool = True,
        has_web: bool = True,
        has_files: bool = True,
        skills_content: str = "",
    ) -> str:
        """Build a focused system prompt for the subagent."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        # Build capability sections dynamically
        can_do = []
        cannot_do = []

        if has_files:
            can_do.append("- Read and write files in the workspace")
        else:
            cannot_do.append("- Read or write files (no file tools available)")

        if has_exec:
            can_do.append("- Execute shell commands")
        else:
            cannot_do.append("- Execute shell commands (no exec tool available)")

        if has_web:
            can_do.append("- Search the web and fetch web pages")
        else:
            cannot_do.append("- Search the web or fetch web pages (no web tools available)")

        can_do.append("- Complete the task thoroughly")

        # These are always true for subagents
        cannot_do.append("- Send messages directly to users (no message tool available)")
        cannot_do.append("- Spawn other subagents")
        cannot_do.append("- Access the main agent's conversation history")

        can_section = "\n".join(can_do)
        cannot_section = "\n".join(cannot_do)

        skills_section = ""
        if skills_content:
            skills_section = f"\n\n## Pre-loaded Skills\n{skills_content}"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
{can_section}

## What You Cannot Do
{cannot_section}

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed){skills_section}

When you have completed the task, provide a clear summary of your findings or actions."""

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
