import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    model: str
    request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    request_options: dict[str, Any] = field(default_factory=dict)


PROVIDERS = {
    "orca": ProviderConfig(
        base_url="https://api.orcarouter.ai/v1",
        model="deepseek/deepseek-v4-flash-0731",
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        request_options={
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1/",
        model="qwen3.8-flash-next",
    ),
}


def _get_non_empty(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    return value if value else default


def load_llm_config(environ: Mapping[str, str] | None = None) -> LLMConfig:
    values = os.environ if environ is None else environ
    provider = _get_non_empty(values, "LLM_PROVIDER", "orca").lower()

    try:
        provider_defaults = PROVIDERS[provider]
    except KeyError as exc:
        available = ", ".join(PROVIDERS)
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Available providers: {available}"
        ) from exc

    request_options = dict(provider_defaults.request_options)
    reasoning_effort = values.get("LLM_REASONING_EFFORT")
    if reasoning_effort:
        request_options["reasoning_effort"] = reasoning_effort

    extra_body = values.get("LLM_EXTRA_BODY")
    if extra_body:
        try:
            parsed_extra_body = json.loads(extra_body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_EXTRA_BODY must be valid JSON: {exc.msg}") from exc
        if not isinstance(parsed_extra_body, dict):
            raise ValueError("LLM_EXTRA_BODY must be a JSON object")
        request_options["extra_body"] = parsed_extra_body

    return LLMConfig(
        provider=provider,
        api_key=values.get("LLM_API_KEY", ""),
        base_url=_get_non_empty(values, "LLM_BASE_URL", provider_defaults.base_url),
        model=_get_non_empty(values, "LLM_MODEL", provider_defaults.model),
        request_options=request_options,
    )
