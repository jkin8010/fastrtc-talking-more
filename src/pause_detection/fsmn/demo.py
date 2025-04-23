#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import sys
import numpy as np
import soundfile
import logging

# 添加src目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入模块
from pause_detection.fsmn.model import FSMNVADModel, FSMNVadOptions

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("FSMN VAD模型演示")
    
    # 初始化模型
    model = FSMNVADModel()
    model_path = model.model_path
    
    # 方法1: 直接处理完整音频文件
    print("\n方法1: 处理完整音频文件")
    wav_file = os.path.join(model_path, "example/vad_example.wav")
    if not os.path.exists(wav_file):
        print(f"示例音频文件不存在: {wav_file}")
        print("尝试使用ModelScope示例音频")
        wav_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"
    
    # 直接使用模型的generate方法处理
    res = model.model.generate(input=wav_file)
    print("检测到的语音片段 (开始时间, 结束时间) 单位为毫秒:")
    print(res)
    
    # 方法2: 使用我们的VAD接口处理
    print("\n方法2: 使用我们的VAD接口处理")
    try:
        if wav_file.startswith("http"):
            # 如果是URL，先尝试下载
            import urllib.request
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                urllib.request.urlretrieve(wav_file, temp_file.name)
                wav_file = temp_file.name
            except Exception as e:
                print(f"下载音频文件失败: {e}")
                return
        
        speech, sample_rate = soundfile.read(wav_file)
        
        # 转换为float32类型
        if speech.dtype != np.float32:
            speech = speech.astype(np.float32)
        
        # 如果是立体声，取第一个通道
        if len(speech.shape) > 1 and speech.shape[1] > 1:
            speech = speech[:, 0]
        
        # 使用我们的VAD接口
        options = FSMNVadOptions()
        duration, chunks = model.vad((sample_rate, speech), options)
        
        print(f"检测到语音总时长: {duration:.2f}秒")
        print("检测到的语音片段 (开始采样点, 结束采样点):")
        for i, chunk in enumerate(chunks):
            start_ms = chunk["start"] * 1000 / sample_rate
            end_ms = chunk["end"] * 1000 / sample_rate
            print(f"片段 {i+1}: [{start_ms:.2f}ms, {end_ms:.2f}ms] - 时长: {(end_ms-start_ms)/1000:.2f}秒")
    
    except Exception as e:
        print(f"处理音频时出错: {e}")
    
    # 方法3: 流式处理
    print("\n方法3: 流式处理演示")
    try:
        speech, sample_rate = soundfile.read(wav_file)
        
        # 转换为float32类型
        if speech.dtype != np.float32:
            speech = speech.astype(np.float32)
        
        # 如果是立体声，取第一个通道
        if len(speech.shape) > 1 and speech.shape[1] > 1:
            speech = speech[:, 0]
            
        # FSMN VAD使用200ms的窗口大小
        chunk_size = 200  # ms
        chunk_stride = int(chunk_size * sample_rate / 1000)
        cache = {}
        
        # 计算总块数
        total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)
        print(f"将音频分成 {total_chunk_num} 个块进行处理，每块时长 {chunk_size}ms")
        
        # 避免FunASR模型请求进度条输出
        os.environ["DISABLE_FUNASR_PROGRESS"] = "1"
        
        for i in range(total_chunk_num):
            # 提取当前块
            speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
            is_final = i == total_chunk_num - 1
            
            # 使用流式处理
            try:
                res = model.model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    disable_pbar=True,
                )
                
                # 只打印非空结果
                if len(res[0]["value"]):
                    print(f"块 {i+1}/{total_chunk_num} 检测结果: {res}")
            except Exception as e:
                print(f"处理块 {i+1} 时出错: {e}")
                if i > 10:  # 如果已经处理了10个块，就停止继续尝试
                    print("已处理足够的块，跳过剩余部分")
                    break
    
    except Exception as e:
        print(f"流式处理时出错: {e}")

if __name__ == "__main__":
    main() 