from typing import Literal, Optional, Protocol, Tuple, Union
from functools import lru_cache
from collections.abc import AsyncGenerator, Generator
import os
import numpy as np
from numpy.typing import NDArray
import torch
from ChatTTS import Chat
import torchaudio
import librosa
import soundfile as sf
from logging import getLogger

logger = getLogger(__name__)

# 设置PyTorch CUDA内存分配配置以避免碎片化
torch.cuda.empty_cache()
if torch.cuda.is_available():
    # 设置内存分配策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_audio(path: str, target_sr: int) -> NDArray[np.float32]:
    """加载音频文件并确保返回float32类型数据"""
    audio_data, file_sr = sf.read(path)
    if file_sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=file_sr, target_sr=target_sr)
    # 确保返回float32类型
    return audio_data.astype(np.float32)

class TTSOptions:
    """
    Options for Text-to-Speech synthesis.
    """
    stream: bool = False
    lang: Optional[str] = None
    skip_refine_text: bool = False
    refine_text_only: bool = False
    use_decoder: bool = True
    do_text_normalization: bool = True
    do_homophone_replacement: bool = False
    params_refine_text: Optional[Chat.RefineTextParams] = None
    params_infer_code: Optional[Chat.InferCodeParams] = None

    def __init__(self):
        # 初始化必要的参数
        self.params_refine_text = Chat.RefineTextParams()
        self.params_refine_text.prompt = ""  # 添加空的 prompt
        self.params_infer_code = Chat.InferCodeParams()

class TTSModel(Protocol):
    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32 | np.int16]]: 
        pass

    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32 | np.int16]], None]: 
        pass
    
    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32 | np.int16]], None, None]:
        pass

SCRIPT_DIR = os.path.dirname(__file__)
           
class ChatModelTTS(TTSModel):
    """
    A Text-to-Speech model using Hugging Face's CosyVoice model.
    Implements the FastRTC TTSModel protocol.

    Attributes:
        model_id: The Hugging Face model ID
        device: The device to run inference on ('cpu', 'cuda', 'mps')
        dtype: Data type for model weights (float16, float32)
    """

    MODEL_OPTIONS = Literal[
        "2Noise/ChatTTS",
    ]

    def __init__(
        self,
        model: MODEL_OPTIONS = "2Noise/ChatTTS",
        device: str = "cpu",
        voice: str = "zhitian_emo",
    ):
        """
        Initialize the Sambert TTS model.

        Args:
            model: Model to use, options:
                - "ModelM/ChatTTS-ModelScope": ChatTTS模型
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            voice: Voice to use, options:
                - zhitian_emo: 知天（情感）
                - zhizhe_emo: 知哲（情感）
                - zhibei_emo: 知贝（情感）
                - zhiya_emo: 知雅（情感）
                - zhiling_emo: 知灵（情感）
                - zhimeng_emo: 知萌（情感）
        """
        self.model_id = model
        self.device = device
        self.voice = voice
        self.model_path = os.path.join(".huggingface", self.model_id)
        self.model: Chat | None = None
        self.default_speaker_embedding = None
       
        # Load the model
        self._load_model()
        
    def _load_model(self): 
        self.model = Chat()
        if not self.model.has_loaded():
            project_dir = os.getcwd()
            local_path = os.path.join(project_dir, ".huggingface", self.model_id)
            logger.info(f"Loading model from {local_path}")
            self.model.download_models(
                source="local",
                custom_path=local_path,
            )
            self.model.load(
                source="local",
                custom_path=local_path,
                device=self.device,
            )
            self.default_speaker_embedding = torch.load(os.path.join(SCRIPT_DIR, "default_speaker_embedding.pt"))
            
        else:
            logger.info("Model already loaded")
        
    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32 | np.int16]]:
        if options is None:
            options = TTSOptions()
            
        # --- Add this section to set a default speaker ---
        # Replace "YOUR_SPEAKER_EMBEDDING_HERE" with the actual embedding string
        # You can get a random one using self.model.sample_random_speaker()
        # For example: default_speaker_sample = self.model.sample_random_speaker()
        if options.params_infer_code is None:
            options.params_infer_code = Chat.InferCodeParams()
        
        options.params_infer_code.spk_emb = self.default_speaker_embedding
        # --- End of added section ---
        
        logger.info("Text input: %s", str(text))

        # audio seed
        if options.params_infer_code and options.params_infer_code.manual_seed is not None:
            torch.manual_seed(options.params_infer_code.manual_seed)

        # text seed for text refining
        if options.params_refine_text:
            text = self.model.infer(
                text=text, skip_refine_text=False, refine_text_only=True
            )
            logger.info(f"Refined text: {text}")
        else:
            # no text refining
            text = text

        logger.info("Start voice inference.")
        wavs = self.model.infer(
            text=text,
            stream=options.stream,
            lang=options.lang,
            skip_refine_text=options.skip_refine_text,
            use_decoder=options.use_decoder,
            do_text_normalization=options.do_text_normalization,
            do_homophone_replacement=options.do_homophone_replacement,
            params_infer_code=options.params_infer_code,
            params_refine_text=options.params_refine_text,
        )
        logger.info("Inference completed.")

        logger.info(f"Wav list: {wavs}")
        
        if options.stream:
            # 如果是流式模式，直接返回生成器
            return 24000, wavs
        else:
            # 非流式模式，合并所有音频片段
            if isinstance(wavs, (list, tuple)):
                tts_speech = np.concatenate(wavs, axis=0)
            else:
                tts_speech = wavs
            
            # 确保数据类型正确
            if hasattr(tts_speech, 'dtype') and tts_speech.dtype != np.float32:
                tts_speech = tts_speech.astype(np.float32)
            
            # 确保是单通道音频
            if tts_speech.ndim > 1:
                if tts_speech.shape[0] == 3:  # 如果是3通道
                    # 取第一个通道
                    tts_speech = tts_speech[0]
                    logger.info("Converted 3-channel audio to mono by taking first channel")
                else:
                    # 如果是其他多通道情况，取平均值
                    tts_speech = np.mean(tts_speech, axis=0)
                    logger.info("Converted multi-channel audio to mono by averaging")
            
            # 确保是一维数组
            tts_speech = tts_speech.squeeze()
            logger.info(f"Final audio shape: {tts_speech.shape}, dtype: {tts_speech.dtype}")
              
            return 24000, tts_speech
    
    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32 | np.int16]], None]:
        if options is None:
            options = TTSOptions()
        # 强制设置 stream 为 True 以便 tts 返回生成器
        options.stream = True
            
        logger.info(f"Start voice inference {text}.")
        sample_rate, wav_generator = self.tts(text, options)

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
            
        sample_rate, wav_generator = self.tts(text, options)

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
         
            yield sample_rate, wav_chunk
            
    def __del__(self):
        if hasattr(self, 'model') and self.model is not None:
            self.model.unload()
            self.model = None
        
        # 主动清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
      

# For simpler imports
@lru_cache
def get_tts_model(
    model_name: str = "2Noise/ChatTTS",
    device: str = "cpu",
) -> TTSModel:
    """
    Helper function to easily get an STT model instance with warm-up.

    Args:
        model_name: Name of the model to use
        
    Returns:
        A warmed-up TTSModel instance
    """
    m = ChatModelTTS(model=model_name, device=device)
    options = TTSOptions()
    options.params_refine_text = Chat.RefineTextParams()
    options.params_refine_text.prompt = ""  # 添加空的 prompt
    options.params_infer_code = Chat.InferCodeParams()
    m.tts("Hello, world!", options)
    return m
     
     
if __name__ == "__main__":
    m = get_tts_model()
    sample_rate, audio_data = m.tts("你好，我是老六，一个会自由行走的六足机器人，我可以跟随你走，也可以跟你聊天哦")
    
    # 将音频数据转换为正确维度的 torch tensor
    audio_tensor = torch.from_numpy(audio_data).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # [samples] -> [1, samples]
    
    # 调整音频速度（通过压缩时长）
    speed_factor = 1.25  # 提速到1.25倍
    target_length = int(audio_tensor.shape[1] / speed_factor)
    audio_tensor_faster = torch.nn.functional.interpolate(
        audio_tensor.unsqueeze(1),
        size=target_length,
        mode='linear',
        align_corners=False
    ).squeeze(1)
    
    # 保存原始音频和加速后的音频
    torchaudio.save("test_original.wav", audio_tensor, sample_rate)
    torchaudio.save("test_faster.wav", audio_tensor_faster, sample_rate)
    logger.info(f"原始音频已保存到 test_original.wav，采样率：{sample_rate}Hz")
    logger.info(f"加速音频已保存到 test_faster.wav，采样率：{sample_rate}Hz，速度：{speed_factor}x")
