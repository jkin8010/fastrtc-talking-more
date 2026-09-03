import pytest

from llm.config import load_llm_config


def test_orca_is_default():
    config = load_llm_config({"LLM_API_KEY": "test-key"})
    assert config.provider == "orca"
    assert config.base_url == "https://api.orcarouter.ai/v1"
    assert config.model == "deepseek/deepseek-v4-flash-0731"
    assert config.api_key == "test-key"


def test_deepseek_defaults_include_thinking_options():
    config = load_llm_config({"LLM_PROVIDER": "deepseek", "LLM_API_KEY": "key"})
    assert config.model == "deepseek-v4-flash"
    assert config.request_options == {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }


def test_unified_variables_override_provider_defaults():
    config = load_llm_config(
        {
            "LLM_PROVIDER": "ollama",
            "LLM_API_KEY": "key",
            "LLM_BASE_URL": "http://custom/v1",
            "LLM_MODEL": "custom-model",
            "LLM_REASONING_EFFORT": "low",
            "LLM_EXTRA_BODY": '{"foo": "bar"}',
        }
    )
    assert config.base_url == "http://custom/v1"
    assert config.model == "custom-model"
    assert config.request_options == {
        "reasoning_effort": "low",
        "extra_body": {"foo": "bar"},
    }


def test_invalid_extra_body_is_rejected():
    with pytest.raises(ValueError, match="LLM_EXTRA_BODY must be valid JSON"):
        load_llm_config({"LLM_EXTRA_BODY": "not-json"})


def test_unknown_provider_lists_available_names():
    with pytest.raises(ValueError, match="orca.*deepseek.*ollama"):
        load_llm_config({"LLM_PROVIDER": "unknown"})
