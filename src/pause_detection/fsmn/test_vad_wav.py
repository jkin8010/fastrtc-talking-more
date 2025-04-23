#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import soundfile
import logging
import torch
from pathlib import Path
from pause_detection.fsmn.model import get_fsmn_vad_model, FSMNVadOptions
from text_to_speech.megatts.model import get_tts_model
from speech_to_text.funasr.model import get_stt_model
from openai import OpenAI
from fastrtc import ReplyOnPause

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio_file(input_file, stt_model, tts_model, fsmn_vad_model):
    """处理音频文件，使用VAD、STT和TTS模型"""
    logger.info(f"处理音频文件: {input_file}")

    # 读取WAV文件
    try:
        audio, sample_rate = soundfile.read(input_file)
        logger.info(f"音频采样率: {sample_rate}Hz, 形状: {audio.shape}, 时长: {len(audio)/sample_rate:.2f}秒")
        
        # 转换为float32类型
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 如果是立体声，取第一个通道
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio[:, 0]
            
        # 使用FSMN VAD模型检测语音片段
        options = FSMNVadOptions()
        duration, chunks = fsmn_vad_model.vad((sample_rate, audio), options)
        
        logger.info(f"VAD检测到语音总时长: {duration:.2f}秒")
        logger.info(f"VAD检测到{len(chunks)}个语音片段:")
        
        for i, chunk in enumerate(chunks):
            start_ms = chunk["start"] * 1000 / sample_rate
            end_ms = chunk["end"] * 1000 / sample_rate
            logger.info(f"片段 {i+1}: [{start_ms:.2f}ms, {end_ms:.2f}ms] - 时长: {(end_ms-start_ms)/1000:.2f}秒")
            
            # 提取语音片段
            chunk_audio = audio[chunk["start"]:chunk["end"]]
            
            # 使用STT模型转换语音为文本
            text = stt_model.stt((sample_rate, chunk_audio))
            logger.info(f"STT结果: {text}")
            
            if text.strip():
                # 使用TTS模型将文本转换回语音
                logger.info(f"使用TTS生成回复...")
                output_path = Path(input_file).with_suffix('').name + f"_response_{i+1}.wav"
                
                # 这里模拟LLM回复，实际应用中替换为真实LLM调用
                llm_response = f"这是对片段{i+1}「{text}」的回复。"
                
                # 使用TTS生成语音回复
                sample_rate, audio_response = tts_model.tts(llm_response)
                
                # 保存回复音频
                soundfile.write(output_path, audio_response, sample_rate)
                logger.info(f"回复已保存至: {output_path}")

    except Exception as e:
        logger.error(f"处理音频文件时出错: {e}", exc_info=True)

def test_pause_detection(input_file, fsmn_vad_model):
    """测试ReplyOnPause的工作方式"""
    logger.info(f"测试ReplyOnPause暂停检测: {input_file}")
    
    # 读取WAV文件
    try:
        audio, sample_rate = soundfile.read(input_file)
        logger.info(f"音频采样率: {sample_rate}Hz, 形状: {audio.shape}, 时长: {len(audio)/sample_rate:.2f}秒")
        
        # 转换为float32类型
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 如果是立体声，取第一个通道
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio[:, 0]
        
        # 创建一个简单的回调函数，显示检测到的暂停
        def on_pause(audio):
            logger.info("检测到暂停!")
            # 返回音频长度和时间点
            duration_s = len(audio) / sample_rate
            logger.info(f"收到的音频长度: {duration_s:.2f}秒")
            return iter([])
        
        # 创建ReplyOnPause处理器
        pause_handler = ReplyOnPause(
            fn=on_pause,
            model=fsmn_vad_model,
            can_interrupt=True
        )
        
        # 直接调用vad模型测试参数是否已修改
        options = FSMNVadOptions()
        logger.info(f"VAD参数: silence_threshold={options.silence_threshold}, min_rms_threshold={options.min_rms_threshold}")
        
        # 使用完整音频进行VAD分析
        duration, chunks = fsmn_vad_model.vad((sample_rate, audio), options)
        logger.info(f"VAD分析结果: 检测到{len(chunks)}个语音片段, 总时长: {duration:.2f}秒")
        
        # 模拟音频流，按200ms的块处理音频
        chunk_ms = 200
        chunk_samples = int(sample_rate * chunk_ms / 1000)
        
        # 记录处理过程
        logger.info(f"使用流式处理，将音频分成{len(audio) // chunk_samples + 1}个块")
        
        # 初始化音频缓冲区和状态
        audio_buffer = np.array([], dtype=np.float32)
        
        # 按块处理音频
        for i in range(0, len(audio), chunk_samples):
            end = min(i + chunk_samples, len(audio))
            chunk = audio[i:end]
            
            # 添加到缓冲区
            audio_buffer = np.concatenate([audio_buffer, chunk])
            
            # 每处理5个块(1秒)检查一次是否有暂停
            if (i // chunk_samples) % 5 == 4 or end == len(audio):
                logger.info(f"块 {i//chunk_samples + 1}/{len(audio)//chunk_samples + 1}, 处理缓冲区长度: {len(audio_buffer)/sample_rate:.3f}秒")
                
                # 将缓冲区传递给VAD模型进行分析
                duration, vad_chunks = fsmn_vad_model.vad((sample_rate, audio_buffer), options)
                
                if len(vad_chunks) > 0:
                    logger.info(f"检测到{len(vad_chunks)}个语音片段")
                else:
                    logger.info("未检测到语音片段，可能是暂停")
                    # 如果未检测到语音，视为暂停，调用回调函数
                    if len(audio_buffer) > 0:
                        logger.info("调用暂停处理回调")
                        on_pause((sample_rate, audio_buffer))
                
                # 清空缓冲区，开始新的检测
                audio_buffer = np.array([], dtype=np.float32)
                
    except Exception as e:
        logger.error(f"测试ReplyOnPause时出错: {e}")
        import traceback
        logger.debug(f"异常堆栈: {traceback.format_exc()}")

def main():
    parser = argparse.ArgumentParser(description='测试语音端点检测 (VAD) 使用WAV文件输入')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入WAV文件路径')
    parser.add_argument('--mode', '-m', type=str, choices=['full', 'vad'], default='full', 
                        help='模式: full(完整处理) 或 vad(仅测试暂停检测)')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式，显示更多日志')
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        # 为所有下级logger设置级别
        for log_name in ['pause_detection', 'pause_detection.fsmn.model']:
            logging.getLogger(log_name).setLevel(logging.DEBUG)
        logger.info("已启用调试模式")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 加载模型
    fsmn_vad_model = get_fsmn_vad_model()
    
    # 打印默认VAD参数
    options = FSMNVadOptions()
    logger.info(f"VAD默认参数: silence_threshold={options.silence_threshold}, min_rms_threshold={options.min_rms_threshold}, use_default_speech={options.use_default_speech}")
    
    # 根据模式运行不同的处理
    if args.mode == 'vad':
        # 仅测试暂停检测
        test_pause_detection(args.input, fsmn_vad_model)
    else:
        # 完整处理流程
        stt_model = get_stt_model()
        tts_model = get_tts_model(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # 处理音频文件
        process_audio_file(args.input, stt_model, tts_model, fsmn_vad_model)

if __name__ == "__main__":
    main() 