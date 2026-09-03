# FastRTC 中文大模型对话

[![Powered by OrcaRouter](https://img.shields.io/badge/Powered_by-OrcaRouter-2563eb)](https://www.orcarouter.ai/ref/ref_32a70c018a6828cde46f)

基于 FastRTC、FunASR、MegaTTS 和大语言模型服务的实时语音对话应用。

## 功能特点

- 🎙️ 实时语音对话：支持实时语音输入和输出
- 🤖 智能对话：默认基于 OrcaRouter（deepseek/deepseek-v4-flash-0731）大语言模型
- 🗣️ 语音识别：使用 FunASR 进行中文语音识别
- 🔊 语音合成：使用 MegaTTS/ChatTTS 进行中文语音合成
- 🌐 WebRTC 支持：基于 FastRTC 实现实时音视频通信

## 依赖项

- [FastRTC](https://github.com/gradio-app/fastrtc)：实时音视频通信框架
- [FunASR](https://github.com/modelscope/FunASR)：中文语音识别模型
- [MegaTTS](https://github.com/bytedance/MegaTTS3)：字节跳动的智能语音合成模型
- [ChatTTS](https://github.com/2noise/ChatTTS)：中文语音合成模型
- [ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker)：ChatTTS 说话人模型
- [OrcaRouter](https://www.orcarouter.ai/)：提供默认的大语言模型服务

## 安装说明

1. 克隆项目并安装依赖：

```bash
git clone https://github.com/jkin8010/fastrtc-talking-more.git
cd fastrtc-talking-more
```

macOS 安装 MegaTTS 文本归一化依赖所需的 OpenFST：

```bash
brew install openfst
export CPPFLAGS="-I$(brew --prefix openfst)/include"
export LDFLAGS="-L$(brew --prefix openfst)/lib"
```

如果使用 Apple Silicon，也可以使用以下变量让 Pynini 的编译器找到 OpenFST：

```bash
export CPLUS_INCLUDE_PATH="$(brew --prefix openfst)/include"
export LIBRARY_PATH="$(brew --prefix openfst)/lib"
```

安装 Python 依赖：

```bash
uv sync
```

项目中的 MegaTTS 运行时依赖 `openai-whisper` 提供的 `whisper` 模块，以及
`wetextprocessing` 提供的 `tn` 文本归一化模块。当前依赖版本已针对 Python 3.12
和 OpenFST 1.8.4 做了兼容配置：

```text
openai-whisper==20250625
wetextprocessing>=1.2.0
pynini>=2.1.7
```

安装完成后，可以使用以下命令检查关键模块：

```bash
uv run python -c "import whisper; from tn.chinese.normalizer import Normalizer; print('whisper/tn import: OK')"
```

2. 配置 `.env`：

复制环境变量模板：

```bash
cp .env.example .env
```

编辑项目根目录下的 `.env` 文件。所有 LLM Provider 都使用同一组 `LLM_*` 配置项，默认使用 OrcaRouter：

```ini
# 国内镜像
HF_ENDPOINT="https://hf-mirror.com"
# OrcaRouter API Key 获取地址：
# https://api.orcarouter.ai/ref/ref_32a70c018a6828cde46f
LLM_PROVIDER="orca"
LLM_API_KEY="sk-orca-你的密钥"
LLM_BASE_URL="https://api.orcarouter.ai/v1"
LLM_MODEL="deepseek/deepseek-v4-flash-0731"
LLM_REASONING_EFFORT=""
LLM_EXTRA_BODY=""

# 使用 DeepSeek 官方 API 时，只需修改以下统一配置：
# LLM_PROVIDER="deepseek"
# LLM_API_KEY="你的 DeepSeek API Key"
# LLM_BASE_URL="https://api.deepseek.com"
# LLM_MODEL="deepseek-v4-flash"
# LLM_REASONING_EFFORT="high"
# LLM_EXTRA_BODY='{"thinking":{"type":"enabled"}}'

# 使用本地 Ollama 时，只需修改以下统一配置：
# LLM_PROVIDER="ollama"
# LLM_API_KEY="ollama"
# LLM_BASE_URL="http://localhost:11434/v1/"
# LLM_MODEL="qwen3.8-flash-next"
```

将 `LLM_API_KEY` 替换为实际密钥。切换 Provider 时只需要修改 `LLM_PROVIDER`、`LLM_BASE_URL` 和 `LLM_MODEL`，不需要新增 Provider 专属环境变量。

新增 OpenAI-compatible Provider 时，只需在 `src/llm/config.py` 的 `PROVIDERS` 注册表中添加一条 `ProviderConfig`，无需修改启动流程或新增 Provider 专属环境变量。

3. 启动服务：

```bash
uv run start
```

首次启动时，MegaTTS 会从 [Hugging Face](https://huggingface.co/ByteDance/MegaTTS3)
自动下载模型到 `.huggingface/ByteDance/MegaTTS3`。如果已经手动下载模型，
可以在 `.env` 中指定模型目录：

```ini
MEGATTS_CHECKPOINT_PATH="/path/to/MegaTTS3"
```

如果日志中出现 `Repo ByteDance/MegaTTS3 not exists`，说明仍在使用旧版本代码，
请先同步最新代码并重新执行 `uv sync --reinstall-package megatts3`。

## 使用说明

1. 访问 `http://localhost:7860` 打开 Web 界面
2. 点击"开始对话"按钮
3. 允许浏览器访问麦克风
4. 开始语音对话

## 注意事项

- 确保已安装所有依赖项
- macOS 用户需要先安装 OpenFST，再执行 `uv sync`，否则 `pynini` 可能无法编译
- 如果出现 `No module named 'whisper'` 或 `No module named 'tn'`，请重新执行 `uv sync`
- 确保有足够的系统资源运行模型
- 建议使用支持 WebRTC 的现代浏览器

### 排查“有文字回复但没有声音”

LLM 回复成功不代表 TTS 已经完成。请按日志顺序检查：

1. `Preparing MegaTTS prompt` 和 `MegaTTS prompt preparation completed`：确认提示音预处理完成。
2. `Starting MegaTTS inference` 和 `MegaTTS inference completed`：确认模型推理完成并返回 WAV 数据。
3. `Decoded TTS audio` 和 `Finished TTS audio stream`：确认音频非空、非静音，并已切成音频块交给 FastRTC。

如果日志停在 `Language detected`、`Preparing MegaTTS prompt` 或 `Starting MegaTTS inference`，通常是 MegaTTS 在 CPU 上运行较慢，并不是 LLM 没有回复。等待对应的完成日志；生产环境建议使用 CUDA。程序会自动选择 CUDA，其次选择 Apple MPS，最后使用 CPU。

如果出现 `MegaTTS returned an empty WAV file` 或 `MegaTTS returned silent or invalid audio`，请检查 MegaTTS 模型目录、提示音文件，以及模型与当前 PyTorch/设备的兼容性。Apple Silicon 上如果出现 `generated invalid audio before normalization`，程序会自动使用 MPS 的 float32 推理；仍失败时可在支持 CUDA 的环境运行，或将日志中的设备信息一并提交排查。

FunASR 集成测试需要下载并加载外部模型，默认不会在普通 `pytest` 中执行。确认网络和模型缓存可用后，可运行：

```bash
RUN_MODEL_TESTS=1 uv run pytest src/speech_to_text/funasr/test_model.py -v
```

## 相关项目

- [FastRTC](https://github.com/gradio-app/fastrtc)：实时音视频通信框架
- [FunASR](https://github.com/modelscope/FunASR)：中文语音识别模型
- [MegaTTS](https://github.com/bytedance/MegaTTS3)：字节跳动的智能语音合成模型
- [ChatTTS](https://github.com/2noise/ChatTTS)：中文语音合成模型
- [ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker)：ChatTTS 说话人模型

## 许可证

MIT License
