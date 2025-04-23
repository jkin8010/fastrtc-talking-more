from typing import Literal, Optional, Protocol, Tuple, Union
from functools import lru_cache
import os
import sys
from pathlib import Path
import click
import numpy as np
from numpy.typing import NDArray
from funasr import AutoModel
from logging import getLogger
from .debug_utils import save_debug_audio
import librosa # 导入 librosa

logger = getLogger(__name__)

class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


class FunASRSTT(STTModel):
    """
    A Speech-to-Text model using Hugging Face's distil-whisper model.
    Implements the FastRTC STTModel protocol.

    Attributes:
        model_id: The Hugging Face model ID
        device: The device to run inference on ('cpu', 'cuda', 'mps')
        dtype: Data type for model weights (float16, float32)
    """

    MODEL_OPTIONS = Literal[
        "paraformer-zh",
    ]

    def __init__(
        self,
        model: MODEL_OPTIONS = "paraformer-zh",
        vad_model: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the Distil-Whisper STT model.

        Args:
            model: Model size/variant to use
            vad_model: VAD model name to use
            debug_mode: 是否启用调试模式
        """
        self.model_id = model
        self.vad_model = vad_model or "fsmn-vad"
        self.debug_mode = debug_mode

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the model and processor from Hugging Face."""
        
        self.model = AutoModel(
            model=self.model_id, 
            # vad_model=self.vad_model, 
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model="ct-punc", 
            # spk_model="cam++",
        )

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        """
        Transcribe audio to text using FunASR model, resampling if necessary.

        Args:
            audio: Tuple of (sample_rate, audio_data)
                  where audio_data is a numpy array of int16 or float32

        Returns:
            Transcribed text as string
        """
        sample_rate, audio_np = audio
        target_sample_rate = 16000 # FunASR 模型期望的采样率
        
        # 如果开启调试模式，保存原始输入数据
        if self.debug_mode:
            save_debug_audio(audio_np, sample_rate, "input_raw", 
                            {"model": self.model_id, "vad_model": self.vad_model})
        
        logger.info(f"Input audio shape: {audio_np.shape}, dtype: {audio_np.dtype}, sample_rate: {sample_rate}")
        # logger.info(f"Audio min: {np.min(audio_np)}, max: {np.max(audio_np)}, mean: {np.mean(audio_np)}")
        
        # --- 音频预处理 --- 
        
        # 1. 确保音频是单通道的
        if audio_np.ndim > 1:
            logger.info(f"Multi-channel audio detected with shape {audio_np.shape}")
            if len(audio_np.shape) == 2 and audio_np.shape[0] == 1: # 处理 (1, n) 形状
                audio_np = audio_np.squeeze()
                logger.info("Squeezed audio from (1, n) to (n,)")
            elif len(audio_np.shape) == 2 and audio_np.shape[1] < audio_np.shape[0]:  # 可能是[samples, channels]格式
                logger.info("Likely [samples, channels] format, taking mean across channels")
                audio_np = np.mean(audio_np, axis=1)
            elif len(audio_np.shape) == 2 and audio_np.shape[0] < audio_np.shape[1]:  # 可能是[channels, samples]格式
                logger.info("Likely [channels, samples] format, taking mean across channels")
                audio_np = np.mean(audio_np, axis=0)
            else:
                logger.warning("Complex multi-dimensional audio, attempting to flatten")
                audio_np = audio_np.flatten()
        
        # 确保是一维数组
        audio_np = audio_np.squeeze()
        logger.info(f"After channel processing - shape: {audio_np.shape}, dtype: {audio_np.dtype}")
        
        # 2. 处理数据类型，转换为 float32
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
            logger.info("Converted int16 to float32 and normalized by 32768.0")
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
            logger.info("Converted int32 to float32 and normalized by 2147483648.0")
        elif audio_np.dtype == np.uint8:
            audio_np = (audio_np.astype(np.float32) - 128) / 128.0
            logger.info("Converted uint8 to float32 and normalized")
        elif audio_np.dtype == np.float64:
            audio_np = audio_np.astype(np.float32)
            logger.info("Converted float64 to float32")
        elif audio_np.dtype != np.float32:
             logger.warning(f"Unsupported audio dtype {audio_np.dtype}, attempting conversion to float32")
             audio_np = audio_np.astype(np.float32)
        
        # 检查并归一化 float32 数据 (如果需要)
        if np.max(np.abs(audio_np)) > 1.0:
            logger.info(f"Float32 audio with values outside [-1, 1] range, max abs value: {np.max(np.abs(audio_np))}")
            max_val = np.max(np.abs(audio_np))
            if max_val > 1e-6: # 避免除以太小的值
                audio_np = audio_np / max_val
                logger.info(f"Normalized float32 audio by max value {max_val}")
            else:
                 logger.warning("Max absolute value is very small, skipping normalization")

        # 3. 重采样到目标采样率 (16kHz)
        if sample_rate != target_sample_rate:
            logger.info(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz")
            # librosa.resample 期望 float 输入
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=target_sample_rate)
            logger.info(f"Resampled audio shape: {audio_np.shape}")
            current_sample_rate = target_sample_rate # 更新当前采样率
        else:
            current_sample_rate = sample_rate

        # 4. 检查无效值
        if np.isnan(audio_np).any() or np.isinf(audio_np).any():
            logger.warning("Audio contains NaN or Inf values! Replacing with zeros.")
            audio_np = np.nan_to_num(audio_np)
        
        # Final stats after processing
        logger.info(f"Final audio for model - shape: {audio_np.shape}, dtype: {audio_np.dtype}, sample_rate: {current_sample_rate}")
        # logger.info(f"Final audio min: {np.min(audio_np)}, max: {np.max(audio_np)}, mean: {np.mean(audio_np)}")

        # 如果开启调试模式，保存处理后的数据
        if self.debug_mode:
            save_debug_audio(audio_np, current_sample_rate, "processed", 
                            {"model": self.model_id, "vad_model": self.vad_model})

        # --- 运行 FunASR 模型 --- 
        try:
            result = self.model.generate(
                input=audio_np,
                sampling_rate=current_sample_rate, # 使用处理后的采样率
                batch_size_s=300, 
                batch_size_threshold_s=60, 
                hotword='小宇',
            )
            logger.info(f"STT result: {result}")
            
            if len(result) > 0 and "text" in result[0]:
                return result[0]["text"]
            else:
                logger.warning("STT result is empty or does not contain 'text' key.")
                return ""
        except Exception as e:
            logger.error(f"Error during model.generate: {e}", exc_info=True)
            return ""


# For simpler imports
@lru_cache
def get_stt_model(
    model_name: str = "paraformer-zh",
    debug_mode: bool = False,
) -> STTModel:
    """
    Helper function to easily get an STT model instance with warm-up.

    Args:
        model_name: Name of the model to use
        debug_mode: 是否开启调试模式

    Returns:
        A warmed-up STTModel instance
    """
    m = FunASRSTT(model=model_name, debug_mode=debug_mode)
    
    return m
