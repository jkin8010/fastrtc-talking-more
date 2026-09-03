from llm.client import create_llm_client
from llm.config import LLMConfig


def test_create_llm_client_uses_config(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "llm.client.OpenAI",
        lambda **kwargs: calls.update(kwargs) or "client",
    )
    config = LLMConfig(
        "deepseek", "key", "https://api.deepseek.com", "deepseek-v4-flash",
        {"reasoning_effort": "high"},
    )

    result = create_llm_client(config)

    assert result == "client"
    assert calls == {"api_key": "key", "base_url": "https://api.deepseek.com"}
