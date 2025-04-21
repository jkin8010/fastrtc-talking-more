# FastRTC 中文大模型对话示例

基于 FastRTC、FunASR、ChatTTS 和 Qwen2.5 的实时语音对话示例。

## 功能特点

- 🎙️ 实时语音对话：支持实时语音输入和输出
- 🤖 智能对话：基于 Ollama（Qwen2.5） 大语言模型
- 🗣️ 语音识别：使用 FunASR 进行中文语音识别
- 🔊 语音合成：使用 ChatTTS 进行中文语音合成
- 🌐 WebRTC 支持：基于 FastRTC 实现实时音视频通信

## 依赖项

- [FastRTC](https://github.com/gradio-app/fastrtc)：实时音视频通信框架
- [FunASR](https://github.com/modelscope/FunASR)：中文语音识别模型
- [ChatTTS](https://github.com/2noise/ChatTTS)：中文语音合成模型
- [ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker)：ChatTTS 说话人模型
- Qwen2.5：通义千问 2.5 大语言模型

## 安装说明

1. 克隆项目并安装依赖：

```bash
git clone https://github.com/jkin8010/fastrtc-zh-demo.git
cd fastrtc-zh-demo
uv sync
```

2. 配置环境变量：

```bash
# 国内镜像
export HF_ENDPOINT="https://hf-mirror.com"
# 非必要
export OLLAMA_API_KEY="ollama"
export OLLAMA_API_URL="http://localhost:11434/v1/"
```

3. 启动服务：

```bash
uv run start
```

## 使用说明

1. 访问 `http://localhost:7860` 打开 Web 界面
2. 点击"开始对话"按钮
3. 允许浏览器访问麦克风
4. 开始语音对话

## 注意事项

- 确保已安装所有依赖项
- 确保有足够的系统资源运行模型
- 建议使用支持 WebRTC 的现代浏览器

## 相关项目

- [FastRTC](https://github.com/gradio-app/fastrtc)：实时音视频通信框架
- [FunASR](https://github.com/modelscope/FunASR)：中文语音识别模型
- [ChatTTS](https://github.com/2noise/ChatTTS)：中文语音合成模型
- [ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker)：ChatTTS 说话人模型

## 许可证

MIT License
