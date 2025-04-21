import numpy as np
import soundfile as sf
import os
import time
from pathlib import Path
import logging
import json
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class AudioDebugger:
    """用于调试音频数据问题的工具类"""
    
    def __init__(self, debug_dir: str = "audio_debug"):
        """
        初始化调试器
        
        Args:
            debug_dir: 调试文件保存的目录
        """
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        self.counter = 0
    
    def save_audio_data(self, 
                        audio_data: np.ndarray, 
                        sample_rate: int, 
                        source: str = "unknown",
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        保存音频数据到文件，用于调试
        
        Args:
            audio_data: 音频数据数组
            sample_rate: 采样率
            source: 音频来源标识
            metadata: 附加元数据
            
        Returns:
            保存的文件路径
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.counter += 1
        
        # 创建文件名
        filename_base = f"{timestamp}_{self.counter:03d}_{source}"
        audio_path = self.debug_dir / f"{filename_base}.wav"
        
        # 保存音频数据的基本信息
        audio_info = {
            "timestamp": timestamp,
            "sample_rate": sample_rate,
            "duration_seconds": len(audio_data)/sample_rate if len(audio_data) > 0 else 0,
            "shape": audio_data.shape if hasattr(audio_data, "shape") else None,
            "dtype": str(audio_data.dtype) if hasattr(audio_data, "dtype") else None,
            "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1] if len(audio_data.shape) > 1 else None,
            "min_value": float(np.min(audio_data)) if len(audio_data) > 0 else None,
            "max_value": float(np.max(audio_data)) if len(audio_data) > 0 else None,
            "mean_value": float(np.mean(audio_data)) if len(audio_data) > 0 else None,
            "std_value": float(np.std(audio_data)) if len(audio_data) > 0 else None,
            "source": source
        }
        
        # 添加附加元数据
        if metadata:
            audio_info.update(metadata)
        
        # 保存元数据
        meta_path = self.debug_dir / f"{filename_base}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(audio_info, f, ensure_ascii=False, indent=2)
        
        # 保存音频
        try:
            # 确保数据是一维或[samples, channels]格式
            if len(audio_data.shape) == 2:
                if audio_data.shape[0] == 1:  # 处理 (1, n) 形状
                    logger.info(f"处理 (1, n) 形状音频 shape={audio_data.shape}, squeezing...")
                    audio_data = audio_data.squeeze()
                elif audio_data.shape[0] > audio_data.shape[1]:
                    # 可能是[channels, samples]格式，需要转置
                    if audio_data.shape[0] <= 8:  # 假设不会有超过8个通道
                        logger.info(f"转置可能的[channels, samples]格式音频 shape={audio_data.shape}")
                        audio_data = audio_data.T
            
            # 处理数据类型
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                # 确保浮点数据在[-1, 1]范围内
                if np.max(np.abs(audio_data)) > 1.0:
                    norm_factor = np.max(np.abs(audio_data))
                    # 避免除以零
                    if norm_factor > 1e-9:
                        audio_data = audio_data / norm_factor
                        logger.info(f"归一化浮点音频数据，除以因子 {norm_factor}")
                    else:
                        logger.warning("音频最大绝对值接近零，跳过归一化")
            
            sf.write(audio_path, audio_data, sample_rate)
            logger.info(f"已保存调试音频到: {audio_path}")
            return str(audio_path)
        except Exception as e:
            logger.error(f"保存音频时出错: {e}")
            # 尝试保存原始数据
            np_path = self.debug_dir / f"{filename_base}_raw.npy"
            try:
                np.save(np_path, audio_data)
                logger.info(f"已保存原始numpy数据到: {np_path}")
            except Exception as ne:
                logger.error(f"保存原始numpy数据时出错: {ne}")
            return str(np_path)

# 创建全局调试器实例，便于在不同地方使用
debugger = AudioDebugger()

def save_debug_audio(audio_data: np.ndarray, 
                    sample_rate: int, 
                    source: str = "unknown",
                    metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    便捷函数，用于保存调试音频
    """
    return debugger.save_audio_data(audio_data, sample_rate, source, metadata) 