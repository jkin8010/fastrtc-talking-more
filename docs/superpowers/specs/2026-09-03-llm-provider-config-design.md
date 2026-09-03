# LLM Provider 配置重构设计

## 背景

当前 `src/main.py` 通过 `if/elif` 分支创建 OrcaRouter、DeepSeek 和 Ollama 客户端。每增加一个 Provider，都需要修改启动逻辑，并为每个 Provider 增加一组带 Provider 前缀的环境变量，配置入口逐渐分散。

本次重构不保留旧的 `ORCA_*`、`DEEPSEEK_*`、`OLLAMA_*` 环境变量兼容。新配置统一使用 `LLM_*` 变量，默认 Provider 仍为 `orca`。

## 目标

- 将 Provider 选择和配置解析从 `main.py` 中移出。
- 用 Provider 注册表替代 Provider 分支逻辑。
- 统一 API Key、Base URL、模型和请求扩展参数的配置入口。
- 保持 OpenAI-compatible Chat Completions 调用方式。
- 支持 OrcaRouter、DeepSeek、Ollama 的现有行为。
- 新增 Provider 时只需注册默认配置，不修改主启动流程。

## 配置设计

统一环境变量如下：

```ini
LLM_PROVIDER="orca"
LLM_API_KEY="..."
LLM_BASE_URL="..."
LLM_MODEL="..."
LLM_REASONING_EFFORT="high"
LLM_EXTRA_BODY='{"thinking":{"type":"enabled"}}'
```

Provider 注册项提供默认值；统一环境变量优先覆盖注册项默认值。`LLM_REASONING_EFFORT` 和 `LLM_EXTRA_BODY` 为可选项，未设置时不传给 SDK。

## 架构

新增 `src/llm/` 模块：

- `config.py`：定义 `ProviderConfig`，维护内置 Provider 注册表，并解析统一环境变量。
- `client.py`：根据解析结果创建 `OpenAI` 客户端及请求选项。
- `__init__.py`：暴露稳定的配置/客户端工厂接口。

`main.py` 只调用 LLM 工厂并将返回的客户端、模型和请求选项传给 `EchoHandler`，不再知道各 Provider 的环境变量或默认参数。

## 内置 Provider

| Provider | 默认 Base URL | 默认模型 | 特殊请求参数 |
| --- | --- | --- | --- |
| `orca` | `https://api.orcarouter.ai/v1` | `deepseek/deepseek-v4-flash-0731` | 无 |
| `deepseek` | `https://api.deepseek.com` | `deepseek-v4-flash` | `reasoning_effort=high`、thinking enabled |
| `ollama` | `http://localhost:11434/v1/` | `qwen3.8-flash-next` | 无 |

## 错误处理

- 未知 Provider：抛出包含可用 Provider 名称的明确错误。
- `LLM_EXTRA_BODY` 不是合法 JSON：抛出包含变量名和解析原因的配置错误。
- API Key 缺失时不在配置解析阶段伪造 Provider 专属默认值；由 SDK 请求阶段返回认证错误。

## 测试与验收

- 单元测试覆盖 Provider 默认值、统一环境变量覆盖、JSON 扩展参数和未知 Provider。
- 验证 `main.py` 不再包含 Provider-specific `if/else` 或 Provider 前缀环境变量读取。
- README 和 `.env.example` 只展示统一配置，并说明新增 Provider 的注册方式。
- 运行 Python 语法检查和项目测试（如环境依赖允许）。
