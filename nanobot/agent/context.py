"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from contextlib import suppress
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.utils.helpers import (
    build_assistant_message,
    current_time_str,
    detect_image_mime,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    _MAX_HISTORY_CHARS = 32_000  # hard cap on recent history section size
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(self, workspace: Path, timezone: str | None = None, disabled_skills: list[str] | None = None):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity(channel=channel)]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        if tool_registry:
            deferred_section = self._build_deferred_tools_section(tool_registry)
            if deferred_section:
                parts.append(deferred_section)

        entries = self.memory.read_unprocessed_history(since_cursor=self.memory.get_last_dream_cursor())
        if entries:
            capped = entries[-self._MAX_RECENT_HISTORY:]
            history_text = "\n".join(
                f"- [{e['timestamp']}] {e['content']}" for e in capped
            )
            history_text = truncate_text(history_text, self._MAX_HISTORY_CHARS)
            parts.append("# Recent History\n\n" + history_text)

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_deferred_tools_section(registry: ToolRegistry) -> str:
        """Build system prompt section listing deferred tools available via resolve_tools."""
        catalog = registry.get_deferred_catalog()
        if not catalog:
            return ""
        lines = [
            "# Deferred Tools",
            "",
            "The following tools are available but their full schemas are not loaded yet.",
            "Call `resolve_tools` with the tool name(s) you need before using them.",
            "You can resolve multiple tools in a single call.",
            "",
        ]
        for entry in catalog:
            lines.append(f"- **{entry['name']}** — {entry['description']}")
        return "\n".join(lines)

    def _get_identity(self, channel: str | None = None) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    _VOICE_CHANNEL_HINT = (
        "CRITICAL: This is a voice channel — your response will be read aloud by a TTS engine. "
        "Keep responses under 30 words — one or two short sentences. "
        "ABSOLUTELY NO markdown: no **, no -, no bullet points, no numbered lists, no links, no code blocks, no headers, no formatting of any kind. "
        "NEVER use emojis, acronyms, or abbreviations — spell them out (say 'kilometres per hour' not 'km/h'). "
        "Avoid special characters, symbols, or anything that doesn't sound natural when spoken. "
        "Write plain spoken English only — exactly as you would say it out loud in conversation. "
        "If listing items, say them in a flowing sentence, not as a list. "
        "Use at most 1 tool call only if essential."
    )

    @staticmethod
    def _build_runtime_context(
        channel: str | None, chat_id: str | None, timezone: str | None = None,
        session_summary: str | None = None, sender_id: str | None = None,
    ) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            display_channel = "voice" if channel in ("alexa", "esphome") else channel
            lines += [f"Channel: {display_channel}", f"Chat ID: {chat_id}"]
        if channel in ("alexa", "esphome"):
            lines.append(ContextBuilder._VOICE_CHANNEL_HINT)
        if sender_id:
            lines += [f"Sender ID: {sender_id}"]
        if session_summary:
            lines += ["", "[Resumed Session]", session_summary]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + ContextBuilder._RUNTIME_CONTEXT_END

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        with suppress(Exception):
            tpl = pkg_files("nanobot") / "templates" / template_path
            if tpl.is_file():
                return content.strip() == tpl.read_text(encoding="utf-8").strip()
        return False

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        session_summary: str | None = None,
        tool_registry: ToolRegistry | None = None,
        sender_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id, self.timezone, session_summary=session_summary, sender_id=sender_id)
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content
        messages = [
            {"role": "system", "content": self.build_system_prompt(skill_names, channel=channel, tool_registry=tool_registry)},
            *history,
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: Any,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
