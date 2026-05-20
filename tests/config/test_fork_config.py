"""Regression tests for fork-specific configuration surfaces."""

from nanobot.config.schema import AlexaConfig, Config


def test_alexa_config_is_available_for_channel_manager() -> None:
    cfg = AlexaConfig()

    assert cfg.enabled is False
    assert cfg.endpoint_path == "/alexa"
    assert cfg.allow_from == ["*"]


def test_alexa_config_accepts_camel_case_channel_settings() -> None:
    cfg = AlexaConfig.model_validate(
        {
            "enabled": True,
            "port": 8444,
            "verifySignatures": True,
            "launchMessage": "I'm listening",
        }
    )

    assert cfg.enabled is True
    assert cfg.port == 8444
    assert cfg.endpoint_path == "/alexa"
    assert cfg.verify_signatures is True
    assert cfg.launch_message == "I'm listening"


def test_gateway_webhook_config_fields_exist() -> None:
    cfg = Config()

    assert cfg.gateway.webhook_secret == ""
    assert cfg.gateway.webhook_channel == ""
    assert cfg.gateway.webhook_chat_id == ""
