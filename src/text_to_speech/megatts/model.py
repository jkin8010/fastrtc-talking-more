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
from fastrtc import AdditionalOutputs

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
    ) -> tuple[int, NDArray[np.float32], AdditionalOutputs]:
        if options is None:
            options = TTSOptions()

        logger.info("Text input: %s", str(text))

        # Detect language
        language_type = classify_language(text)
        if language_type not in ["zh", "zh-cn", "en"]:
            raise ValueError(f"Unsupported language detected: {language_type}")
        
        if language_type == "zh-cn":
            language_type = "zh"
            
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

        # Convert wav bytes to numpy array
        wav_array = np.frombuffer(wav_bytes, dtype=np.float32)
        return 24000, wav_array, AdditionalOutputs()

    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32], AdditionalOutputs], None]:
        if options is None:
            options = TTSOptions()
        # 强制设置 stream 为 True 以便 tts 返回生成器
        options.stream = True
            
        logger.info(f"Start voice inference {text}.")
        sample_rate, wav_generator, additional_outputs = self.tts(text, options)

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
            yield sample_rate, wav_chunk, additional_outputs

    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32], AdditionalOutputs], None, None]:
        if options is None:
            options = TTSOptions()
        # 强制设置 stream 为 True 以便 tts 返回生成器
        options.stream = True
            
        sample_rate, wav_generator, additional_outputs = self.tts(text, options)

        # 直接迭代 tts 返回的生成器
        for wav_chunk in wav_generator:
            # 确保音频块是 float32 类型
            if hasattr(wav_chunk, 'dtype') and wav_chunk.dtype != np.float32:
                wav_chunk = wav_chunk.astype(np.float32)

            # 确保是单声道音频
            if wav_chunk.ndim > 1:
                # 如果是多通道情况，取平均值
                wav_chunk = np.mean(wav_chunk, axis=0)
                logger.debug("Converted multi-channel chunk to mono by averaging in sync stream")
            
            # 确保是一维数组
            wav_chunk = wav_chunk.squeeze()
         
            yield sample_rate, wav_chunk, additional_outputs

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
    return MegaTTSModel(device=device)

if __name__ == "__main__":
    m = get_tts_model()
    sample_rate, audio_data, additional_outputs = m.tts("你好，我是MegaTTS，一个强大的语音合成模型！")
    
    # 保存音频
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_wav(audio_data, os.path.join(project_root, "audio_debug", "output.wav"))
    logger.info(f"音频已保存到 output.wav，采样率：{sample_rate}Hz")