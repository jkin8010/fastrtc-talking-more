# LLM Provider Configuration Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Provider-specific branching and environment variables with a registry-driven LLM configuration that supports OrcaRouter, DeepSeek, and Ollama through one stable `LLM_*` interface.

**Architecture:** Add a focused `src/llm/` package. `config.py` owns Provider defaults and environment parsing, while `client.py` creates the OpenAI-compatible client and request options. `main.py` consumes one factory result and does not contain Provider-specific environment lookups.

**Tech Stack:** Python 3.12+, OpenAI SDK, `dataclasses`, `json`, `pytest`, python-dotenv.

**Spec:** `docs/superpowers/specs/2026-09-03-llm-provider-config-design.md`

## Global Constraints

- Do not read or document legacy `ORCA_*`, `DEEPSEEK_*`, or `OLLAMA_*` environment variables.
- Default Provider is `orca`.
- Keep OpenAI-compatible `chat.completions.create(..., stream=True)` behavior.
- Built-in defaults: OrcaRouter `deepseek/deepseek-v4-flash-0731`, DeepSeek `deepseek-v4-flash`, Ollama `qwen3.8-flash-next`.
- DeepSeek defaults include `reasoning_effort="high"` and `extra_body={"thinking": {"type": "enabled"}}`.
- `LLM_EXTRA_BODY` must be valid JSON when present.

---

### Task 1: Add registry-driven LLM configuration

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/config.py`
- Test: `tests/test_llm_config.py`

**Interfaces:**
- Produces `ProviderConfig`, `LLMConfig`, `PROVIDERS`, and `load_llm_config(environ=None) -> LLMConfig`.
- `LLMConfig` contains `provider`, `api_key`, `base_url`, `model`, and `request_options`.

- [ ] **Step 1: Write tests for defaults, overrides, JSON parsing, and unknown Provider**

```python
def test_orca_is_default(monkeypatch):
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
    config = load_llm_config({
        "LLM_PROVIDER": "ollama",
        "LLM_API_KEY": "key",
        "LLM_BASE_URL": "http://custom/v1",
        "LLM_MODEL": "custom-model",
        "LLM_REASONING_EFFORT": "low",
        "LLM_EXTRA_BODY": '{"foo": "bar"}',
    })
    assert config.base_url == "http://custom/v1"
    assert config.model == "custom-model"
    assert config.request_options == {
        "reasoning_effort": "low", "extra_body": {"foo": "bar"}
    }

def test_unknown_provider_lists_available_names():
    with pytest.raises(ValueError, match="orca.*deepseek.*ollama"):
        load_llm_config({"LLM_PROVIDER": "unknown"})
```

- [ ] **Step 2: Run the focused tests and verify they fail because the module is absent**

Run: `uv run pytest tests/test_llm_config.py -q`

- [ ] **Step 3: Implement immutable Provider defaults and environment-driven overrides**

```python
@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    model: str
    request_options: dict[str, Any] = field(default_factory=dict)

PROVIDERS = {
    "orca": ProviderConfig("https://api.orcarouter.ai/v1", "deepseek/deepseek-v4-flash-0731"),
    "deepseek": ProviderConfig(
        "https://api.deepseek.com", "deepseek-v4-flash",
        {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}},
    ),
    "ollama": ProviderConfig("http://localhost:11434/v1/", "qwen3.8-flash-next"),
}
```

The loader must use `LLM_PROVIDER` (default `orca`), `LLM_API_KEY` (default empty), and optional `LLM_BASE_URL`, `LLM_MODEL`, `LLM_REASONING_EFFORT`, and `LLM_EXTRA_BODY`. Parse `LLM_EXTRA_BODY` with `json.loads` and raise `ValueError("LLM_EXTRA_BODY must be valid JSON: ...")` on failure. Copy request options before applying overrides.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `uv run pytest tests/test_llm_config.py -q`

- [ ] **Step 5: Commit the configuration module and tests**

```bash
git add src/llm tests/test_llm_config.py
git commit -m "refactor: add registry-driven LLM configuration"
```

### Task 2: Add the OpenAI-compatible client factory

**Files:**
- Create: `src/llm/client.py`
- Modify: `src/llm/__init__.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Produces `create_llm_client(config: LLMConfig) -> OpenAI`.

- [ ] **Step 1: Write a test patching `OpenAI` and asserting unified config is passed**

```python
def test_create_llm_client_uses_config(monkeypatch):
    calls = {}
    monkeypatch.setattr("llm.client.OpenAI", lambda **kwargs: calls.update(kwargs) or "client")
    config = LLMConfig("deepseek", "key", "https://api.deepseek.com", "deepseek-v4-flash", {"reasoning_effort": "high"})
    result = create_llm_client(config)
    assert result == "client"
    assert calls == {"api_key": "key", "base_url": "https://api.deepseek.com"}
```

- [ ] **Step 2: Run the test and verify it fails because the factory is absent**

Run: `uv run pytest tests/test_llm_client.py -q`

- [ ] **Step 3: Implement the small client factory**

```python
def create_llm_client(config: LLMConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)
```

- [ ] **Step 4: Run the client test and the configuration tests**

Run: `uv run pytest tests/test_llm_config.py tests/test_llm_client.py -q`

- [ ] **Step 5: Commit the client factory**

```bash
git add src/llm tests/test_llm_client.py
git commit -m "refactor: add OpenAI-compatible client factory"
```

### Task 3: Migrate `main.py` to the LLM factory

**Files:**
- Modify: `src/main.py:70-120,240-270`
- Test: `tests/test_main_llm_integration.py`

**Interfaces:**
- `EchoHandler` receives an `LLMConfig` or its `model`/`request_options` without knowing Provider names.
- `main()` calls `load_llm_config()` and `create_llm_client()` exactly once.

- [ ] **Step 1: Add an integration-level test that verifies EchoHandler forwards request options**

```python
class FakeDelta:
    content = "回答。"
    role = function_call = tool_calls = None

class FakeChunk:
    choices = [type("Choice", (), {"delta": FakeDelta()})()]

class FakeSTT:
    def stt(self, audio):
        return "你好"

class FakeTTS:
    def stream_tts_sync(self, text):
        return iter(())

class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter([FakeChunk()])

def test_echo_forwards_configured_request_options():
    completions = FakeCompletions()
    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": completions})()
    })()
    handler = EchoHandler(FakeSTT(), FakeTTS(), client, "deepseek-v4-flash", {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    })
    list(handler.echo((16000, b"audio"), []))
    assert completions.kwargs["model"] == "deepseek-v4-flash"
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["reasoning_effort"] == "high"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
```

- [ ] **Step 2: Run the integration test and verify it fails against the current constructor shape**

Run: `uv run pytest tests/test_main_llm_integration.py -q`

- [ ] **Step 3: Remove Provider-specific `if/elif` and environment reads from `main.py`**

Import the LLM factory, load the unified config, create the client, and pass `config.model` and `config.request_options` to `EchoHandler`. Keep `EchoHandler` generic and preserve `stream=True`.

- [ ] **Step 4: Run focused tests and syntax checks**

Run: `uv run pytest tests/test_llm_config.py tests/test_llm_client.py tests/test_main_llm_integration.py -q`

Run: `python3 -m py_compile src/main.py src/llm/*.py`

- [ ] **Step 5: Commit the main integration**

```bash
git add src/main.py tests/test_main_llm_integration.py
git commit -m "refactor: use provider registry in application startup"
```

### Task 4: Update configuration documentation and validate the project

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_llm_config.py`

- [ ] **Step 1: Replace Provider-specific examples with the unified `LLM_*` configuration**

Document `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, optional `LLM_REASONING_EFFORT`, and optional JSON `LLM_EXTRA_BODY`. Show three short provider examples by changing only the Provider and values, not by introducing prefixed variables.

- [ ] **Step 2: Document the registry extension point**

Explain that adding a built-in OpenAI-compatible Provider means adding one `ProviderConfig` entry in `src/llm/config.py`; Provider-specific request parameters belong in that entry.

- [ ] **Step 3: Search for forbidden legacy environment variables and stale branching**

Run: `rg -n "ORCA_|DEEPSEEK_|OLLAMA_|llm_provider ==|llm_provider ==" src README.md .env.example`

Expected: no matches.

- [ ] **Step 4: Run the complete available test and validation suite**

Run: `uv run pytest -q`

Run: `git diff --check`

- [ ] **Step 5: Commit documentation and final validation changes**

```bash
git add README.md .env.example tests
git commit -m "docs: document unified LLM provider configuration"
```
