from functools import lru_cache
from collections.abc import AsyncGenerator, Generator
import os
import numpy as np
from numpy.typing import NDArray
import torch
from logging import getLogger
from pydub import AudioSegment
from megatts3.tts.utils.audio_utils.io import save_wav, to_wav_bytes
from langdetect import detect as classify_language
from megatts3.tts.infer_cli import MegaTTS3DiTInfer
from text_to_speech import TTSModel, TTSOptions
from modelscope import snapshot_download
import librosa
import io

logger = getLogger(__name__)

# 设置PyTorch CUDA内存分配配置以避免碎片化
torch.cuda.empty_cache()
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HUGGINGFACE_CACHE = os.path.join(SCRIPT_DIR, "..", "..", "..", ".huggingface")
if not os.path.exists(HUGGINGFACE_CACHE):
    os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
    logger.info(f"Created huggingface cache directory at {HUGGINGFACE_CACHE}")
else:
    logger.info(f"Huggingface cache directory already exists at {HUGGINGFACE_CACHE}")

class MegaTTSModel(TTSModel):
    """
    A Text-to-Speech model using MegaTTS.
    Implements the FastRTC TTSModel protocol.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the MegaTTS model.

        Args:
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.device = device
        
        self.checkpoint_path = os.path.join(HUGGINGFACE_CACHE, "ByteDance", "MegaTTS3")
        if not os.path.exists(self.checkpoint_path):
            snapshot_download(
                model_id="ByteDance/MegaTTS3",
                cache_dir=HUGGINGFACE_CACHE,
            )
        logger.info(f"Loading MegaTTS model from {self.checkpoint_path} on {self.device}")
        
        # Initialize the MegaTTS inference instance
        self.infer_instance = MegaTTS3DiTInfer(ckpt_root=self.checkpoint_path)

    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32 | np.int16]]:
        if options is None:
            options = TTSOptions()

        logger.info("Text input: %s", str(text))
        
        # 检查文本是否为空
        if not text or text.strip() == "":
            logger.warning("Empty text input, returning empty audio array")
            return 24000, np.zeros(1024, dtype=np.float32)  # 返回一小段静音

        # Detect language
        try:
            language_type = classify_language(text)
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            language_type = "unknown"
        
        logger.info(f"Language detected: {language_type}")
        if language_type not in ["en", "en-us", "en-gb", "zh", "zh-cn", "zh-tw", "zh-yue", "zh-wyw", "zh-classical", "zh-min-nan", "zh-wuu", "zh-vietnamese"]:
            logger.warning(f"Unsupported language detected: {language_type}")
            return 24000, np.zeros(1024, dtype=np.float32)  # 返回一小段静音
        
        language_type = "zh" if language_type in ["zh-cn", "zh-tw", "zh-yue", "zh-wyw", "zh-classical", "zh-min-nan", "zh-wuu", "zh-vietnamese"] else "en"
        
        language_npy_path = os.path.join(SCRIPT_DIR, f"{language_type}_prompt.npy")
        language_wav_path = os.path.join(SCRIPT_DIR, f"{language_type}_prompt.wav")
        if not os.path.exists(language_npy_path):
            raise FileNotFoundError(f"Language prompt file not found: {language_npy_path}")
        
        with open(language_wav_path, 'rb') as file:
            wav_content = file.read()
        
        resource_context = self.infer_instance.preprocess(wav_content, latent_file=language_npy_path)
        wav_bytes = self.infer_instance.forward(
            resource_context,
            text,
            time_step=options.time_step,
            p_w=options.p_w,
            t_w=options.t_w,
        )

        # 使用 librosa 加载音频数据，确保正确的采样率
        wav_array, sr = librosa.load(io.BytesIO(wav_bytes), sr=24000)
        
        # 音频归一化
        wav_array = wav_array / np.max(np.abs(wav_array))
        wav_array = wav_array * 0.95  # 避免削波
        
        return 24000, wav_array.astype(np.float32)

    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32 | np.int16]], None]:
        if options is None:
            options = TTSOptions()
        # 强制设置 stream 为 True 以便 tts 返回生成器
        options.stream = True
            
        # 检查文本是否为空
        if not text or text.strip() == "":
            logger.warning("Empty text input in async stream_tts, returning silent audio")
            # 返回一段静音
            silence = np.zeros(1024, dtype=np.float32)
            yield 24000, silence
            return
            
        logger.info(f"Start voice inference {text}.")
        sample_rate, wav_generator = self.tts(text, options)

        # 检查是否得到了有效的音频数据
        if isinstance(wav_generator, np.ndarray) and (wav_generator.size == 0 or np.max(np.abs(wav_generator)) < 1e-6):
            logger.warning("No valid audio data returned from TTS in async stream, returning silent audio")
            silence = np.zeros(1024, dtype=np.float32)
            yield 24000, silence
            return

        # 直接迭代 tts 返回的生成器
        for wav_chunk in wav_generator:
            # 确保音频块是 float32 类型
            if hasattr(wav_chunk, 'dtype') and wav_chunk.dtype != np.float32:
                wav_chunk = wav_chunk.astype(np.float32)
            print(f"wav_chunk: {wav_chunk} sample_rate: {sample_rate}")
            # 确保是单声道音频
            if wav_chunk.ndim > 1:
                # 如果是多通道情况，取平均值
                wav_chunk = np.mean(wav_chunk, axis=0)
                logger.debug("Converted multi-channel chunk to mono by averaging in async stream")
            
            # 确保是一维数组
            wav_chunk = wav_chunk.squeeze()
            yield sample_rate, wav_chunk

    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32 | np.int16]], None, None]:
        if options is None:
            options = TTSOptions()
        # 强制设置 stream 为 True 以便 tts 返回生成器
        options.stream = True
        
        # 检查文本是否为空
        if not text or text.strip() == "":
            logger.warning("Empty text input in stream_tts_sync, returning silent audio")
            # 返回一段静音
            silence = np.zeros(1024, dtype=np.float32)
            yield 24000, silence
            return
            
        sample_rate, wav_array = self.tts(text, options)
        
        # 检查是否得到了有效的音频数据
        if wav_array.size == 0 or np.max(np.abs(wav_array)) < 1e-6:
            logger.warning("No valid audio data returned from TTS, returning silent audio")
            silence = np.zeros(1024, dtype=np.float32)
            yield 24000, silence
            return
        
        # 确保音频数据是平面格式，形状为 (samples,)
        if wav_array.ndim > 1:
            wav_array = wav_array.squeeze()
        
        # 确保音频数据是float32格式
        if wav_array.dtype != np.float32:
            wav_array = wav_array.astype(np.float32)
        
        # 将音频数据分成小块
        chunk_size = 1024  # 可以根据需要调整块大小
        for i in range(0, len(wav_array), chunk_size):
            chunk = wav_array[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # 如果最后一个块不够大，用零填充
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            # 将数据重塑为正确的格式 (samples,)
            chunk = chunk.reshape(-1)
            
            # 返回格式：(sample_rate, audio_data)
            yield sample_rate, chunk

@lru_cache
def get_tts_model(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TTSModel:
    """
    Helper function to easily get a MegaTTS model instance.

    Args:
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        
    Returns:
        A MegaTTSModel instance
    """
    logger.info(f"Creating MegaTTSModel on {device}")
    return MegaTTSModel(device=device)
