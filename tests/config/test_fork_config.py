"""Regression tests for fork-specific configuration surfaces."""

from nanobot.config.schema import AlexaConfig, Config


def test_alexa_config_is_available_for_channel_manager() -> None:
    cfg = AlexaConfig()

    assert cfg.enabled is False
    assert cfg.endpoint_path == "/alexa"
    assert cfg.allow_from == ["*"]


def test_gateway_webhook_config_fields_exist() -> None:
    cfg = Config()

    assert cfg.gateway.webhook_secret == ""
    assert cfg.gateway.webhook_channel == ""
    assert cfg.gateway.webhook_chat_id == ""
