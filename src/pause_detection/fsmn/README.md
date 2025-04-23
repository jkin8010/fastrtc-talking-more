# FSMN VAD 模型

本模块提供了基于ModelScope FunASR的FSMN VAD（语音活动检测）实现，支持非流式和流式处理模式。

## 功能特点

- 支持完整音频文件的VAD识别
- 支持流式处理，可以实时检测语音
- 支持自定义检测参数
- 兼容PauseDetectionModel接口

## 使用方法

### 1. 非流式处理（完整音频文件）

```python
from pause_detection import FSMNVADModel, FSMNVadOptions
import soundfile
import numpy as np

# 初始化模型
model = FSMNVADModel()

# 读取音频文件
audio, sample_rate = soundfile.read("audio.wav")
if audio.dtype != np.float32:
    audio = audio.astype(np.float32)

# 设置VAD选项
options = FSMNVadOptions(
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=2000,
    speech_pad_ms=400
)

# 执行VAD
duration, speech_chunks = model.vad((sample_rate, audio), options)

# 处理结果
for i, chunk in enumerate(speech_chunks):
    start_ms = chunk["start"] * 1000 / sample_rate
    end_ms = chunk["end"] * 1000 / sample_rate
    print(f"语音片段 {i+1}: [{start_ms:.2f}ms, {end_ms:.2f}ms] - 时长: {(end_ms-start_ms)/1000:.2f}秒")
```

### 2. 流式处理

```python
from pause_detection import FSMNVADModel
import soundfile
import numpy as np

# 初始化模型
model = FSMNVADModel()

# 读取音频文件
audio, sample_rate = soundfile.read("audio.wav")
if audio.dtype != np.float32:
    audio = audio.astype(np.float32)

# 设置流式处理参数
chunk_size = 200  # ms
chunk_stride = int(chunk_size * sample_rate / 1000)
cache = {}

# 分块处理
for i in range(0, len(audio), chunk_stride):
    speech_chunk = audio[i:i + chunk_stride]
    is_final = i + chunk_stride >= len(audio)
    
    # 流式处理当前块
    res = model.model.generate(
        input=speech_chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        disable_pbar=True,
    )
    
    # 处理结果
    if len(res[0]["value"]):
        print(f"检测结果: {res}")
```

## VAD结果格式说明

流式处理中，VAD结果有以下几种格式：

1. `[[beg1, end1], [beg2, end2], .., [begN, endN]]` - 普通格式，表示完整的语音片段
2. `[[beg, -1]]` - 表示检测到语音开始，但尚未结束
3. `[[-1, end]]` - 表示检测到语音结束，开始时间未知

所有时间单位均为毫秒。

## 选项说明

`FSMNVadOptions` 类提供以下选项：

- `threshold`: 语音阈值，高于此值的概率被视为语音
- `min_speech_duration_ms`: 最小语音持续时间，短于此值的片段会被丢弃
- `max_speech_duration_s`: 最大语音持续时间，超过此值的会被分割
- `min_silence_duration_ms`: 在语音片段末尾等待的最小静音时间，用于分离语音
- `window_size_samples`: 送入VAD模型的音频块大小
- `speech_pad_ms`: 在语音片段两侧添加的填充时间
- `chunk_size`: 流式处理的块大小（毫秒） 