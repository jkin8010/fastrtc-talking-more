import logging
from typing import Literal, Dict, List, Tuple, Optional, Any
import warnings
import click
import os
import soundfile
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from functools import lru_cache
from funasr import AutoModel

from fastrtc.utils import AudioChunk, audio_to_float32
import torch
from pause_detection.protocol import PauseDetectionModel

logger = logging.getLogger(__name__)

@lru_cache
def get_fsmn_vad_model() -> PauseDetectionModel:
    """Returns the VAD model instance and warms it up with dummy data."""
    # Warm up the model with dummy data
    model = FSMNVADModel()
    logger.info(click.style("INFO", fg="green") + ":\t  Warming up VAD model.")
    model.warmup()
    logger.info(click.style("INFO", fg="green") + ":\t  VAD model warmed up.")
    return model


@dataclass
class FSMNVadOptions:
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
        Values other than these may affect model performance!!
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
      chunk_size: Chunk size in ms for streaming processing
      use_default_speech: 如果没有检测到语音，是否使用默认语音段落
      detect_silence: 是否检测静音
      silence_threshold: 音频能量低于此阈值被视为静音
      min_rms_threshold: 音频的RMS能量最小阈值，低于此值认为是静音
      energy_threshold_ratio: 当音频能量超过min_rms_threshold的此倍数时，视为有效语音
      energy_change_threshold: 当能量变化超过此值时，视为能量状态改变
      continuous_silence_chunks: 连续多少个块静音被视为暂停
      max_buffer_duration_s: 最大缓冲区持续时间，超过此时间强制处理缓冲区
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 800  # 增加最小语音持续时间，减少短噪声干扰
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 500  # 减少静音检测时长，更快识别暂停
    window_size_samples: int = 1024
    speech_pad_ms: int = 300  # 减少语音边界填充
    chunk_size: int = 200  # ms, for streaming processing
    use_default_speech: bool = True  # 如果没有检测到语音，使用默认语音段落
    detect_silence: bool = True  # 是否主动检测静音
    silence_threshold: float = 0.08  # 静音能量阈值，降低以提高对低音量输入的敏感度
    min_rms_threshold: float = 0.08  # 最小RMS阈值，降低以处理低能量麦克风输入
    energy_threshold_ratio: float = 1.3  # 当能量超过min_rms_threshold的多少倍时视为语音
    energy_change_threshold: float = 0.005  # 能量变化阈值
    continuous_silence_chunks: int = 2  # 连续多少个块静音被视为暂停
    max_buffer_duration_s: float = 5.0  # 最大缓冲区持续时间，超过此时间强制处理

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

class FSMNVADModel:
    def __init__(self, model: Literal["fsmn-vad", "iic/speech_fsmn_vad_zh-cn-8k-common-pytorch"] = "fsmn-vad", model_revision: str = "v2.0.4", device: Optional[str] = None):
        """Initialize FSMN VAD model.
        
        Args:
            model: Model name or path
            model_revision: Model revision
        """
        try:
            logger.info(f"初始化FSMN VAD模型: {model}, 版本: {model_revision}")
            self.model = AutoModel(model=model, model_revision=model_revision, device=device or "cuda" if torch.cuda.is_available() else "cpu")
            self.model_path = self.model.model_path
            logger.info(f"FSMN VAD模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"初始化FSMN VAD模型失败: {e}")
            import traceback
            logger.debug(f"异常堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"初始化FSMN VAD模型失败: {e}")
            
        self.sampling_rate = 16000
        # 为调试添加计数器
        self._vad_call_count = 0
        self._last_speech_detected = False
        
        # 设置numpy的异常处理
        try:
            # 设置numpy不抛出关于无效值的警告
            np.seterr(all='warn', invalid='warn')
            logger.debug("设置numpy错误处理: all='warn', invalid='warn'")
        except Exception as e:
            logger.warning(f"设置numpy错误处理失败: {e}")
            
        # 记录支持的数据类型
        supported_dtypes = ", ".join([
            "float32", "int16", "float64 (会自动转换为float32)",
            "int32 (会自动标准化到[-1,1]范围)",
            "int64 (会自动标准化到[-1,1]范围)"
        ])
        logger.info(f"支持的音频数据类型: {supported_dtypes}")

    @staticmethod
    def collect_chunks(audio: np.ndarray, chunks: list[AudioChunk]) -> np.ndarray:
        """Collects and concatenates audio chunks."""
        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(
            [audio[chunk["start"] : chunk["end"]] for chunk in chunks]
        )

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        vad_options: FSMNVadOptions,
        **kwargs,
    ) -> list[AudioChunk]:
        """This method is used for splitting long audios into speech chunks using FSMN VAD.

        Args:
            audio: One dimensional float array.
            vad_options: Options for VAD processing.
            kwargs: VAD options passed as keyword arguments for backward compatibility.

        Returns:
            List of dicts containing begin and end samples of each speech chunk.
        """
        # 确保音频长度足够
        if len(audio) < 1600:  # 至少100ms
            logger.warning(f"音频太短: {len(audio)/self.sampling_rate*1000:.2f}ms，不进行VAD处理")
            if vad_options.use_default_speech:
                # 如果设置了use_default_speech，则将整个音频视为语音
                return [{"start": 0, "end": len(audio)}]
            return []

        # 处理不同的音频数据类型
        original_dtype = audio.dtype
        if original_dtype == np.float64:
            logger.debug(f"转换音频数据类型: {original_dtype} -> float32")
            audio = audio.astype(np.float32)
        elif original_dtype not in [np.float32, np.int16]:
            logger.debug(f"转换音频数据类型: {original_dtype} -> float32")
            try:
                if original_dtype == np.int32:
                    audio = (audio / 2147483647).astype(np.float32)
                elif original_dtype == np.int64:
                    audio = (audio / 9223372036854775807).astype(np.float32)
                else:
                    audio = audio.astype(np.float32)
                    # 如果值范围不是[-1,1]，进行归一化
                    max_abs = np.max(np.abs(audio))
                    if max_abs > 1.0:
                        audio = audio / max_abs
            except Exception as e:
                logger.warning(f"转换音频类型失败: {e}")

        # 记录原始音频信息
        audio_rms = np.sqrt(np.mean(np.square(audio)))
        audio_abs_mean = np.mean(np.abs(audio))
        logger.debug(f"音频RMS: {audio_rms:.6f}, 平均绝对值: {audio_abs_mean:.6f}, 长度: {len(audio)/self.sampling_rate:.2f}秒")
        
        # 计算音频能量数据，用于调试
        audio_std = np.std(audio)
        audio_max = np.max(np.abs(audio))
        audio_p50 = np.percentile(np.abs(audio), 50)  # 中位数
        audio_p90 = np.percentile(np.abs(audio), 90)  # 90%分位数
        logger.debug(f"音频统计: 标准差={audio_std:.6f}, 最大值={audio_max:.6f}, 中位数={audio_p50:.6f}, 90%分位数={audio_p90:.6f}")
        
        # 如果开启静音检测且RMS值低于阈值，直接认为是静音
        if vad_options.detect_silence and audio_rms < vad_options.silence_threshold:
            logger.info(f"音频能量很低(RMS={audio_rms:.5f} < {vad_options.silence_threshold})，直接视为静音")
            return []
        
        # 使用傅里叶变换分析音频频谱
        try:
            import numpy.fft as fft
            # 分析音频频谱特性，帮助区分人声和噪音
            spectrum = np.abs(fft.rfft(audio))
            # 计算在语音主频段(300-3400Hz)的能量占比
            freq_bins = fft.rfftfreq(len(audio), 1.0/self.sampling_rate)
            voice_mask = (freq_bins >= 300) & (freq_bins <= 3400)
            voice_energy = np.sum(spectrum[voice_mask]**2)
            total_energy = np.sum(spectrum**2) if np.sum(spectrum**2) > 0 else 1
            voice_ratio = voice_energy / total_energy
            
            logger.debug(f"语音频段能量占比: {voice_ratio:.2%}")
            
            # 如果语音频段能量占比很低，可能是噪音而非语音
            if voice_ratio < 0.3 and audio_rms < vad_options.silence_threshold * 3:
                logger.info(f"语音频段能量占比低({voice_ratio:.2%})且总能量较低，可能是噪音而非语音")
                return []
        except Exception as e:
            # 分析失败时继续常规处理
            logger.debug(f"频谱分析失败: {e}")
        
        # 使用完整音频进行语音识别
        try:
            # 如果音频格式不正确，确保转换为正确的格式
            if audio.dtype != np.float32 and audio.dtype != np.int16:
                logger.debug(f"转换音频格式: {audio.dtype} -> float32")
                audio = audio.astype(np.float32)
                
            # 确保音频值在合理范围内
            if audio.dtype == np.float32 and (np.max(np.abs(audio)) > 10 or np.max(np.abs(audio)) < 0.001):
                logger.debug(f"音频值范围异常, 最大值: {np.max(np.abs(audio))}, 规范化处理")
                # 规范化音频
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
            
            res = self.model.generate(input=audio, disable_pbar=True)
            logger.debug(f"VAD原始检测结果: {res}")
            
            # 检查返回结果格式，确保能处理不同情况
            segments = []
            
            # 分析结果格式
            if isinstance(res, list):
                # 情况1: 直接返回段列表 [[beg1, end1], [beg2, end2], ..]
                if len(res) > 0 and isinstance(res[0], list) and len(res[0]) == 2 and all(isinstance(x, (int, float)) for x in res[0]):
                    segments = res
                # 情况2: 返回字典列表 [{'key': xxx, 'value': [[beg1, end1], ...]}]
                elif len(res) > 0 and isinstance(res[0], dict) and 'value' in res[0]:
                    value = res[0]['value']
                    if isinstance(value, list):
                        segments = value
            elif isinstance(res, dict):
                # 情况3: 返回单个字典 {'key': xxx, 'value': [[beg1, end1], ...]}
                if 'value' in res:
                    segments = res['value']
            
            logger.debug(f"解析后的语音段: {segments}")
            
            # 转换结果格式为AudioChunk
            speech_chunks = []
            for segment in segments:
                if not isinstance(segment, list) or len(segment) != 2:
                    logger.warning(f"无效的VAD段格式: {segment}")
                    continue
                
                # FSMN VAD返回的是毫秒级别的时间戳，需要转换为采样点
                start_sample = int(segment[0] * self.sampling_rate / 1000)
                end_sample = int(segment[1] * self.sampling_rate / 1000)
                
                # 确保结束点不超过音频长度
                end_sample = min(end_sample, len(audio))
                
                # 检查语音段的有效性
                if end_sample <= start_sample:
                    logger.debug(f"无效的语音段: [{start_sample}, {end_sample}]，跳过")
                    continue
                
                # 检查语音段的最小持续时间
                if (end_sample - start_sample) / self.sampling_rate * 1000 < vad_options.min_speech_duration_ms:
                    logger.debug(f"语音段太短: {(end_sample - start_sample) / self.sampling_rate * 1000:.1f}ms < {vad_options.min_speech_duration_ms}ms，跳过")
                    continue
                
                # 分析语音段的能量，确保真实有声
                if end_sample > start_sample:
                    segment_audio = audio[start_sample:end_sample]
                    segment_rms = np.sqrt(np.mean(np.square(segment_audio)))
                    
                    # 若当前段能量过低，可能是误识别，跳过
                    if segment_rms < vad_options.min_rms_threshold:
                        logger.debug(f"语音段能量太低: RMS={segment_rms:.5f} < {vad_options.min_rms_threshold}，跳过")
                        continue
                        
                    speech_chunks.append({"start": start_sample, "end": end_sample})
            
            # 合并相近的语音段
            if len(speech_chunks) > 1:
                speech_chunks.sort(key=lambda x: x["start"])
                merged_chunks = [speech_chunks[0]]
                
                min_gap_samples = int(150 * self.sampling_rate / 1000)  # 150ms的最小间隔
                
                for chunk in speech_chunks[1:]:
                    last_chunk = merged_chunks[-1]
                    # 如果当前段与上一段间隔小于150ms，则合并
                    if chunk["start"] - last_chunk["end"] < min_gap_samples:
                        last_chunk["end"] = chunk["end"]
                    else:
                        merged_chunks.append(chunk)
                
                speech_chunks = merged_chunks
            
            # 如果没有检测到语音，但设置了使用默认语音，且未被静音检测判定为静音
            if not speech_chunks and vad_options.use_default_speech and audio_rms >= vad_options.silence_threshold:
                logger.info(f"未检测到语音，但音频能量足够({audio_rms:.5f})，将整个音频视为语音")
                speech_chunks = [{"start": 0, "end": len(audio)}]
            
            logger.debug(f"VAD检测到 {len(speech_chunks)} 个语音段")
            return speech_chunks
            
        except Exception as e:
            logger.error(f"VAD处理出错: {str(e)}")
            if vad_options.use_default_speech:
                # 出错时使用默认处理
                return [{"start": 0, "end": len(audio)}]
            return []

    def process_in_chunks(
        self, 
        audio: np.ndarray, 
        vad_options: FSMNVadOptions,
        chunk_size: Optional[int] = None,
    ) -> list[AudioChunk]:
        """Process audio in chunks for streaming VAD.
        
        Args:
            audio: Audio data as numpy array
            vad_options: VAD options
            chunk_size: Size of chunks in ms, if None use from options
            
        Returns:
            List of speech chunks
        """
        if chunk_size is None:
            chunk_size = vad_options.chunk_size
        
        # 处理不同的音频数据类型
        original_dtype = audio.dtype
        if original_dtype == np.float64:
            logger.debug(f"流式处理: 转换音频数据类型 {original_dtype} -> float32")
            audio = audio.astype(np.float32)
        elif original_dtype not in [np.float32, np.int16]:
            logger.debug(f"流式处理: 转换音频数据类型 {original_dtype} -> float32")
            try:
                if original_dtype == np.int32:
                    audio = (audio / 2147483647).astype(np.float32)
                elif original_dtype == np.int64:
                    audio = (audio / 9223372036854775807).astype(np.float32)
                else:
                    audio = audio.astype(np.float32)
                    # 如果值范围不是[-1,1]，进行归一化
                    max_abs = np.max(np.abs(audio))
                    if max_abs > 1.0:
                        audio = audio / max_abs
            except Exception as e:
                logger.warning(f"流式处理: 转换音频类型失败: {e}")
                
        chunk_stride = int(chunk_size * self.sampling_rate / 1000)
        cache = {}
        total_chunk_num = int((len(audio) - 1) / chunk_stride + 1)
        
        # 计算全局RMS和其他特征作为参考
        global_rms = np.sqrt(np.mean(np.square(audio)))
        global_abs_mean = np.mean(np.abs(audio))
        global_std = np.std(audio)
        global_p90 = np.percentile(np.abs(audio), 90)  # 90%分位数
        
        logger.debug(f"流式处理 {total_chunk_num} 个音频块，每块 {chunk_size}ms，全局RMS={global_rms:.5f}，平均绝对值={global_abs_mean:.5f}，标准差={global_std:.5f}，90%分位={global_p90:.5f}，数据类型={audio.dtype}")
        
        # 判断音频是否为静音
        is_silence = global_rms < vad_options.silence_threshold
        
        # 如果全局能量极低，直接视为静音
        if vad_options.detect_silence and is_silence:
            logger.info(f"整体音频能量极低(RMS={global_rms:.5f} < {vad_options.silence_threshold})，直接视为静音")
            return []
        
        all_segments = []
        silent_chunk_count = 0
        has_detected_speech = False
        
        # 用于记录连续帧的能量变化
        energy_history = []
        
        for i in range(total_chunk_num):
            speech_chunk = audio[i * chunk_stride : (i + 1) * chunk_stride]
            is_final = i == total_chunk_num - 1
            
            # 确保音频块格式正确
            if speech_chunk.dtype != np.float32 and speech_chunk.dtype != np.int16:
                logger.debug(f"流式处理: 转换音频块 {i+1}/{total_chunk_num} 数据类型 {speech_chunk.dtype} -> float32")
                try:
                    speech_chunk = speech_chunk.astype(np.float32)
                except Exception as e:
                    logger.warning(f"流式处理: 转换音频块数据类型失败: {e}")
            
            # 计算当前块的RMS能量
            chunk_rms = np.sqrt(np.mean(np.square(speech_chunk)))
            chunk_abs_mean = np.mean(np.abs(speech_chunk))
            
            # 更新能量历史
            energy_history.append(chunk_rms)
            if len(energy_history) > 5:  # 保留最近5帧的能量历史
                energy_history.pop(0)
            
            # 计算能量变化趋势
            energy_trend = 0
            if len(energy_history) >= 3:
                # 计算最近3帧的平均能量和之前帧的平均能量比较
                recent_avg = sum(energy_history[-3:]) / 3
                previous_avg = sum(energy_history[:-3]) / max(1, len(energy_history) - 3)
                energy_trend = recent_avg - previous_avg
            
            chunk_is_silent = chunk_rms < vad_options.silence_threshold
            
            # 低于阈值标记为静音，但要考虑是否是能量下降趋势
            if chunk_is_silent:
                silent_chunk_count += 1
                # 如果连续多个块都是静音，可能是真正的暂停
                if silent_chunk_count >= vad_options.continuous_silence_chunks and has_detected_speech:  # 至少600ms的静音
                    logger.debug(f"流式处理: 检测到连续{silent_chunk_count}个静音块，当前块RMS={chunk_rms:.5f}")
                    # 不处理这些静音块，直接跳过
                    continue
            else:
                silent_chunk_count = 0  # 重置静音计数
            
            try:
                # 标准化处理音频块，确保格式正确
                if speech_chunk.dtype != np.float32 and speech_chunk.dtype != np.int16:
                    speech_chunk = speech_chunk.astype(np.float32)
                
                # 确保音频值在合理范围内
                if speech_chunk.dtype == np.float32 and np.max(np.abs(speech_chunk)) > 0:
                    max_val = np.max(np.abs(speech_chunk))
                    if max_val > 10 or max_val < 0.001:
                        # 规范化音频
                        speech_chunk = speech_chunk / (max_val + 1e-8) * 0.9
                
                # 判断当前块是否为静音
                if vad_options.detect_silence and chunk_is_silent:
                    logger.debug(f"流式处理: 块 {i+1}/{total_chunk_num} 能量低 (RMS={chunk_rms:.5f} < {vad_options.silence_threshold})，视为静音")
                    continue
                
                # 捕获可能的数据类型异常
                try:
                    res = self.model.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=is_final,
                        chunk_size=chunk_size,
                        disable_pbar=True,
                    )
                except TypeError as e:
                    logger.warning(f"流式处理: 模型输入类型错误: {e}，尝试转换数据类型")
                    if isinstance(speech_chunk, np.ndarray):
                        speech_chunk = speech_chunk.astype(np.float32)
                        # 确保值范围在[-1, 1]
                        if np.max(np.abs(speech_chunk)) > 1.0:
                            speech_chunk = speech_chunk / np.max(np.abs(speech_chunk))
                    # 重试模型生成
                    res = self.model.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=is_final,
                        chunk_size=chunk_size,
                        disable_pbar=True,
                    )
                
                # 处理结果
                chunk_has_speech = False
                if res and isinstance(res, list) and len(res) > 0 and 'value' in res[0] and len(res[0]['value']):
                    segments = res[0]['value']
                    for segment in segments:
                        if not isinstance(segment, list) or len(segment) != 2:
                            continue
                            
                        # 转换毫秒时间戳为样本索引，并加上当前块的偏移
                        start_ms, end_ms = segment
                        
                        # 处理特殊情况: [beg, -1] 或 [-1, end]
                        if start_ms == -1:  # [-1, end] 格式
                            # 结束时间点，开始时间未知
                            end_sample = int(end_ms * self.sampling_rate / 1000) + (i * chunk_stride)
                            start_sample = i * chunk_stride  # 假设从当前块开始
                        elif end_ms == -1:  # [beg, -1] 格式
                            # 开始时间点，结束时间未知
                            start_sample = int(start_ms * self.sampling_rate / 1000) + (i * chunk_stride)
                            end_sample = (i + 1) * chunk_stride  # 假设到当前块结束
                        else:  # 普通 [beg, end] 格式
                            start_sample = int(start_ms * self.sampling_rate / 1000) + (i * chunk_stride)
                            end_sample = int(end_ms * self.sampling_rate / 1000) + (i * chunk_stride)
                        
                        # 确保不超出音频范围
                        start_sample = max(0, min(start_sample, len(audio)))
                        end_sample = max(0, min(end_sample, len(audio)))
                        
                        # 检查语音段长度是否合理
                        segment_duration_ms = (end_sample - start_sample) / self.sampling_rate * 1000
                        if segment_duration_ms < vad_options.min_speech_duration_ms:
                            logger.debug(f"流式处理: 语音段太短 ({segment_duration_ms:.1f}ms)，跳过")
                            continue
                        
                        # 检查语音段能量是否足够
                        if end_sample > start_sample:
                            segment_audio = audio[start_sample:end_sample]
                            segment_rms = np.sqrt(np.mean(np.square(segment_audio)))
                            if segment_rms < vad_options.min_rms_threshold:
                                logger.debug(f"流式处理: 语音段能量太低 (RMS={segment_rms:.5f})，跳过")
                                continue
                            
                            all_segments.append({"start": start_sample, "end": end_sample})
                            chunk_has_speech = True
                            has_detected_speech = True
                
                # 如果能量够大但未检测到语音，根据能量创建语音段
                if not chunk_has_speech and not chunk_is_silent and (i > 0 or not is_silence):
                    # 如果模型未检测到语音，但音频能量足够，尝试根据能量创建语音段
                    if chunk_rms > vad_options.min_rms_threshold * 1.5:
                        logger.debug(f"流式处理: 块 {i+1}/{total_chunk_num} 能量足够 (RMS={chunk_rms:.5f})但未检测到语音，根据能量创建语音段")
                        start_sample = i * chunk_stride
                        end_sample = min((i + 1) * chunk_stride, len(audio))
                        all_segments.append({"start": start_sample, "end": end_sample})
                        has_detected_speech = True
            except Exception as e:
                logger.error(f"处理音频块 {i+1}/{total_chunk_num} 时出错: {e}")
                import traceback
                logger.debug(f"异常堆栈: {traceback.format_exc()}")
        
        # 如果没有检测到语音，但设置了使用默认语音且未被判定为静音
        if not all_segments and vad_options.use_default_speech and not is_silence:
            logger.info(f"流式处理未检测到语音，但音频能量足够({global_rms:.5f})，将整个音频视为语音")
            all_segments = [{"start": 0, "end": len(audio)}]
        
        # 合并重叠段
        if all_segments:
            all_segments.sort(key=lambda x: x["start"])
            merged_segments = [all_segments[0]]
            
            # 150ms的最小间隔
            min_gap_samples = int(150 * self.sampling_rate / 1000)
            
            for segment in all_segments[1:]:
                last = merged_segments[-1]
                # 如果当前段与上一段重叠或非常接近，则合并
                if segment["start"] <= last["end"] + min_gap_samples:
                    last["end"] = max(last["end"], segment["end"])
                else:
                    merged_segments.append(segment)
            
            # 过滤掉太短的片段
            min_speech_samples = int(vad_options.min_speech_duration_ms * self.sampling_rate / 1000)
            filtered_segments = []
            for segment in merged_segments:
                if segment["end"] - segment["start"] >= min_speech_samples:
                    filtered_segments.append(segment)
                else:
                    logger.debug(f"合并后语音段太短 ({(segment['end']-segment['start'])/self.sampling_rate*1000:.1f}ms)，丢弃")
            
            logger.debug(f"流式VAD检测到 {len(filtered_segments)} 个语音段")
            return filtered_segments
        
        logger.debug("流式VAD未检测到语音段")
        return []

    def warmup(self):
        """预热模型，避免首次使用时的延迟"""
        try:
            logger.info("预热FSMN VAD模型开始...")
            
            # 创建多种不同数据类型的测试数据
            sample_rate = 16000
            duration = 2  # 2秒
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # 创建包含不同数据类型的测试序列
            test_data = []
            
            # 1. 生成440Hz的语音信号 (float32)
            freq = 440  # A4音符
            audio_float32 = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            # 添加一些随机噪声
            audio_float32 = audio_float32 + 0.01 * np.random.randn(len(audio_float32)).astype(np.float32)
            # 确保振幅在[-1, 1]之间
            audio_float32 = np.clip(audio_float32, -1, 1)
            test_data.append(("float32", audio_float32))
            
            # 2. 生成float64版本
            audio_float64 = audio_float32.astype(np.float64)
            test_data.append(("float64", audio_float64))
            
            # 3. 生成int16版本
            audio_int16 = (audio_float32 * 32767).astype(np.int16)
            test_data.append(("int16", audio_int16))
            
            # 4. 静音数据
            silence = np.zeros(sample_rate, dtype=np.float32)
            test_data.append(("silence", silence))
            
            # 依次对每种数据类型进行预热
            for data_type, audio_data in test_data:
                try:
                    logger.debug(f"预热使用 {data_type} 数据类型, 形状={audio_data.shape}, 数据类型={audio_data.dtype}")
                    self.vad((sample_rate, audio_data), None)
                except Exception as e:
                    logger.warning(f"使用 {data_type} 数据类型预热失败: {e}")
                    # 失败后继续尝试其他数据类型
                    
            logger.info("FSMN VAD模型预热完成")
        except Exception as e:
            logger.error(f"VAD模型预热错误: {e}")
            import traceback
            logger.debug(f"异常堆栈: {traceback.format_exc()}")
            
            # 使用基本的预热方式作为后备
            try:
                logger.info("使用备用方法尝试预热...")
                for _ in range(3):
                    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1秒的静音
                    self.vad((16000, dummy_audio), None)
                logger.info("备用预热完成")
            except Exception as e2:
                logger.error(f"备用预热也失败: {e2}")
                # 即使预热失败，仍然继续，可能会在第一次调用时有延迟

    def vad(
        self,
        audio: tuple[int, NDArray[np.float32] | NDArray[np.int16]],
        options: None | FSMNVadOptions,
    ) -> tuple[float, list[AudioChunk]]:
        """执行语音活动检测

        Args:
            audio: 采样率和音频数据的元组
            options: VAD选项

        Returns:
            处理后的音频时长和语音片段列表
        """
        # 记录调用次数，用于调试
        self._vad_call_count += 1
        call_id = self._vad_call_count
        sampling_rate, audio_ = audio
        
        try:
            # 处理不同的音频数据类型，确保可以处理常见的数据类型
            if isinstance(audio_, np.ndarray):
                if audio_.dtype == np.float64:
                    logger.debug(f"VAD调用 #{call_id}: 转换音频数据类型 {audio_.dtype} -> float32")
                    audio_ = audio_.astype(np.float32)
                elif audio_.dtype not in [np.float32, np.int16]:
                    logger.debug(f"VAD调用 #{call_id}: 转换音频数据类型 {audio_.dtype} -> float32")
                    try:
                        audio_ = audio_.astype(np.float32)
                    except Exception as e:
                        logger.warning(f"VAD调用 #{call_id}: 转换音频类型失败: {e}")
            
            audio_seconds = len(audio_) / sampling_rate
            # 计算原始未处理音频的RMS
            original_audio_rms = np.sqrt(np.mean(np.square(audio_)))
            # 使用绝对值来避免极低信号导致的异常值
            audio_abs_mean = np.mean(np.abs(audio_))
            
            logger.debug(f"VAD调用 #{call_id}: 音频长度={audio_seconds:.2f}秒, RMS={original_audio_rms:.5f}, 平均绝对值={audio_abs_mean:.5f}, 形状={audio_.shape}, 数据类型={audio_.dtype}")
            
            # 直接静音检测 - 如果音频能量非常低，可能是静音
            if options and options.detect_silence and original_audio_rms < options.silence_threshold:
                logger.info(f"VAD调用 #{call_id}: 检测到静音 (RMS={original_audio_rms:.5f} < {options.silence_threshold})")
                self._last_speech_detected = False
                # 返回空的语音段
                return 0.0, []
            
            # 确保在传递给audio_to_float32之前，音频数据是支持的类型
            try:
                audio_ = audio_to_float32(audio_)
            except TypeError:
                logger.warning(f"VAD调用 #{call_id}: audio_to_float32转换失败，尝试手动转换")
                if isinstance(audio_, np.ndarray):
                    if audio_.dtype == np.float64:
                        audio_ = audio_.astype(np.float32)
                    elif audio_.dtype == np.int32:
                        audio_ = (audio_ / 2147483647).astype(np.float32)
                    elif audio_.dtype == np.int64:
                        audio_ = (audio_ / 9223372036854775807).astype(np.float32)
                    else:
                        # 尝试通用转换
                        audio_ = audio_.astype(np.float32)
                        # 如果值范围不是[-1,1]，进行归一化
                        max_abs = np.max(np.abs(audio_))
                        if max_abs > 1.0:
                            audio_ = audio_ / max_abs
            
            sr = self.sampling_rate
            if sr != sampling_rate:
                try:
                    import librosa  # type: ignore
                except ImportError as e:
                    raise RuntimeError(
                        "Applying the VAD filter requires librosa if the input sampling rate is not 16000hz"
                    ) from e
                audio_ = librosa.resample(audio_, orig_sr=sampling_rate, target_sr=sr)

            if not options:
                options = FSMNVadOptions()
                
            # 如果音频太短，直接处理而不进行复杂操作
            if len(audio_) < sr * 0.1:  # 小于100ms
                if options.use_default_speech and original_audio_rms > options.min_rms_threshold:
                    speech_chunks = [{"start": 0, "end": len(audio_)}]
                    logger.debug(f"VAD调用 #{call_id}: 音频太短 ({len(audio_)/sr*1000:.1f}ms)，视为语音")
                else:
                    speech_chunks = []
                    logger.debug(f"VAD调用 #{call_id}: 音频太短 ({len(audio_)/sr*1000:.1f}ms)，且能量低，视为静音")
            else:
                # 处理音频，根据音频长度决定是否使用流式处理
                if len(audio_) > sr * 10:  # 如果音频超过10秒，使用流式处理
                    logger.debug(f"VAD调用 #{call_id}: 使用流式处理 ({len(audio_)/sr:.1f}秒)")
                    speech_chunks = self.process_in_chunks(audio_, options)
                else:
                    logger.debug(f"VAD调用 #{call_id}: 使用批处理 ({len(audio_)/sr:.1f}秒)")
                    speech_chunks = self.get_speech_timestamps(audio_, options)
                
            # 检查是否有语音被检测到
            has_speech = len(speech_chunks) > 0
            
            # 计算检测到的语音占比
            if has_speech:
                speech_samples = sum(chunk["end"] - chunk["start"] for chunk in speech_chunks)
                speech_ratio = speech_samples / len(audio_)
                logger.debug(f"VAD调用 #{call_id}: 语音占比 {speech_ratio:.2%}")
                
                # 如果语音占比过高（超过90%），可能是误检
                if speech_ratio > 0.9 and options.detect_silence and original_audio_rms < options.silence_threshold * 2:
                    logger.info(f"VAD调用 #{call_id}: 语音占比过高({speech_ratio:.2%})但能量低({original_audio_rms:.5f})，可能是误检，标记为静音")
                    has_speech = False
                    speech_chunks = []
            
            if has_speech != self._last_speech_detected:
                logger.info(f"VAD调用 #{call_id}: {'检测到语音' if has_speech else '未检测到语音'}")
                self._last_speech_detected = has_speech
            
            # 如果没有检测到语音，但有足够的能量，而且之前没有被认为是静音，再考虑作为语音处理
            # 这里添加一个检查，确保前面没有因为options.silence_threshold被判定为静音
            if not has_speech and options.use_default_speech and original_audio_rms > options.min_rms_threshold and original_audio_rms >= options.silence_threshold and len(audio_) > sr * 0.5:
                speech_chunks = [{"start": 0, "end": len(audio_)}]
                logger.info(f"VAD调用 #{call_id}: 未检测到语音但有足够能量({original_audio_rms:.5f})，视为语音")
            
            processed_audio = self.collect_chunks(audio_, speech_chunks)
            duration_after_vad = processed_audio.shape[0] / sr
            
            logger.debug(f"VAD调用 #{call_id}: 处理后音频长度={duration_after_vad:.2f}秒, 检测到{len(speech_chunks)}个语音段")
            return duration_after_vad, speech_chunks
        except Exception as e:
            import math
            import traceback

            logger.error(f"VAD调用 #{call_id} 异常: {str(e)}")
            exec = traceback.format_exc()
            logger.debug(f"VAD调用 #{call_id} 异常堆栈: {exec}")
            
            # 出错时，如果开启了默认语音选项，则视为语音
            if options and options.use_default_speech:
                speech_chunks = [{"start": 0, "end": len(audio_)}]
                return len(audio_) / sr, speech_chunks
            return math.inf, []

    def __call__(self, audio_chunk, cache=None, is_final=False, chunk_size=None):
        """流式处理单个音频块
        
        Args:
            audio_chunk: 音频数据块
            cache: 缓存状态
            is_final: 是否为最后一块
            chunk_size: 块大小 (ms)
            
        Returns:
            检测结果和更新后的缓存
        """
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
            
        if len(audio_chunk.shape) == 1:
            audio_chunk = np.expand_dims(audio_chunk, 0)
        
        # 记录音频特征，用于调试    
        audio_rms = np.sqrt(np.mean(np.square(audio_chunk)))
        logger.debug(f"VAD __call__: 音频形状={audio_chunk.shape}, RMS={audio_rms:.5f}, is_final={is_final}")
            
        # 调用模型进行处理
        try:
            result = self.model.generate(
                input=audio_chunk,
                cache=cache or {},
                is_final=is_final,
                chunk_size=chunk_size or 200,
                disable_pbar=True,
            )
            
            # 检查结果格式
            if result and len(result) > 0:
                if isinstance(result[0], dict) and 'value' in result[0]:
                    segments = result[0]['value']
                    if segments:
                        logger.debug(f"VAD __call__: 检测到语音段 {segments}")
            
            # 返回结果和更新后的缓存
            return result, cache
        except Exception as e:
            logger.error(f"VAD __call__ 异常: {e}")
            # 返回空结果，保留缓存
            return [{'key': 'error', 'value': []}], cache
