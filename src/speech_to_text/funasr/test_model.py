import numpy as np
import soundfile as sf
from speech_to_text.funasr.model import get_stt_model
import argparse
import os
import json
import logging
from pathlib import Path # 导入Path

# 获取脚本所在的目录
SCRIPT_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_funasr():
    # 使用相对于脚本目录的路径
    audio_path = os.path.join(SCRIPT_DIR, "..", "..", "asr_example_zh.wav")
    audio_data, sample_rate = sf.read(audio_path)
    
    # 获取模型实例
    model = get_stt_model(debug_mode=True)
    
    # 进行语音识别
    result = model.stt((sample_rate, audio_data))
    
    print(f"音频文件: {audio_path}")
    print(f"采样率: {sample_rate} Hz")
    print(f"音频长度: {len(audio_data)/sample_rate:.2f} 秒")
    print(f"识别结果: {result}")

def analyze_audio_file(file_path):
    """分析音频文件的属性，帮助调试"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    try:
        audio_data, sample_rate = sf.read(file_path)
        
        audio_info = {
            "文件路径": file_path,
            "采样率": sample_rate,
            "时长(秒)": len(audio_data)/sample_rate,
            "形状": audio_data.shape,
            "数据类型": str(audio_data.dtype),
            "通道数": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            "最小值": float(np.min(audio_data)),
            "最大值": float(np.max(audio_data)),
            "平均值": float(np.mean(audio_data)),
            "标准差": float(np.std(audio_data))
        }
        
        print(json.dumps(audio_info, indent=2, ensure_ascii=False))
        
        # 运行识别
        model = get_stt_model(debug_mode=True)
        result = model.stt((sample_rate, audio_data))
        print(f"识别结果: {result}")
        
    except Exception as e:
        print(f"分析音频文件时出错: {e}")

def save_debug_audio(audio_data, sample_rate, output_path="debug_audio.wav"):
    """保存音频数据到文件，用于调试"""
    try:
        sf.write(output_path, audio_data, sample_rate)
        print(f"已保存调试音频到: {output_path}")
    except Exception as e:
        print(f"保存调试音频时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试FunASR语音识别模型')
    parser.add_argument('--test', action='store_true', help='运行基本测试')
    parser.add_argument('--analyze', type=str, help='分析指定的音频文件')
    
    args = parser.parse_args()
    
    if args.test:
        test_funasr()
    elif args.analyze:
        analyze_audio_file(args.analyze)
    else:
        test_funasr() 