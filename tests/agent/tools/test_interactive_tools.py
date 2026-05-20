"""Tests for interactive user tools."""

from nanobot.agent.tools.context import RequestContext, ToolContext
from nanobot.agent.tools.interactive import AskUserChoiceTool
from nanobot.bus.queue import MessageBus


def test_interactive_tool_create_wires_bus_callback() -> None:
    bus = MessageBus()

    tool = AskUserChoiceTool.create(ToolContext(config=None, workspace=".", bus=bus))

    assert isinstance(tool, AskUserChoiceTool)
    assert tool._send_callback == bus.publish_outbound


def test_interactive_tool_accepts_request_context_and_resolves_pending_future() -> None:
    tool = AskUserChoiceTool()
    tool.set_context(RequestContext(channel="telegram", chat_id="chat-1", message_id="msg-1"))

    assert tool._default_channel == "telegram"
    assert tool._default_chat_id == "chat-1"
    assert tool._default_message_id == "msg-1"

    class _Future:
        def __init__(self) -> None:
            self.value = None

        def done(self) -> bool:
            return False

        def set_result(self, value: str) -> None:
            self.value = value

    fut = _Future()
    tool._pending["telegram:chat-1"] = fut  # type: ignore[assignment]

    assert tool.resolve("telegram", "chat-1", "yes") is True
    assert fut.value == "yes"
    assert "telegram:chat-1" not in tool._pending
