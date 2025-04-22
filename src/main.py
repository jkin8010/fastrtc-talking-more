import os
import logging
import sys
import re
import time
import torch
import asyncio
import numpy as np
from fastrtc import (ReplyOnPause, Stream)
from text_to_speech.megatts.model import get_tts_model, TTSModel
from speech_to_text.funasr.model import get_stt_model, STTModel
from openai import OpenAI
from logging import getLogger

logger = getLogger(__name__)


max_threads = max(min(torch.get_num_threads(), torch.get_num_interop_threads()) - 2, 1)
print(max_threads)

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)

# 设置 PyTorch 线程数
torch.set_num_threads(max_threads)
torch.set_num_interop_threads(1)


class EchoHandler:
    def __init__(self, stt_model, tts_model, llm_client):
        self.stt_model: STTModel = stt_model
        self.tts_model: TTSModel = tts_model
        self.llm_client: OpenAI = llm_client
        self.logger = getLogger(__name__ + ".EchoHandler")

    def echo(self, audio):
        """
        Handles audio input: performs STT, calls LLM, performs TTS, and yields audio output.
        Relies on fastrtc's can_interrupt=True for handling interruptions.
        """
        self.logger.info("Detected pause (or stop word + pause), starting echo function.")
        try:
            prompt = self.stt_model.stt(audio)
            print(f"STT result: {prompt}")

            if not prompt or prompt.strip() == "":
                self.logger.warning("STT result is empty or whitespace, skipping LLM call.")
                return iter([])

            response_stream = self.llm_client.chat.completions.create(
                model="qwen2.5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                stream=True
            )

            # Process LLM stream and collect full response
            buffer_str = ""
            punctuation_marks = ["。", "！", "？", "；", ". ", "! ", "? ", "; ", "\n\n"]
            min_segment_length = 2  # 最小字符数要求
            
            self.logger.info("Starting to receive LLM stream.")
            print("LLM response stream: ", end="", flush=True)
            
            for chunk in response_stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    text_chunk = delta.content
                    print(text_chunk, end="", flush=True)  # 打印即时反馈
                    buffer_str += text_chunk
                    
                    # 检查buffer中是否有结束标点，且长度足够
                    for mark in punctuation_marks:
                        if mark in buffer_str and len(buffer_str.split(mark)[0]) > min_segment_length:
                            # 分离出要处理的部分
                            parts = buffer_str.split(mark, 1)
                            segment_to_process = parts[0] + mark
                            buffer_str = parts[1]  # 剩余部分保留在buffer中
                            
                            self.logger.debug(f"Processing segment during streaming: '{segment_to_process}'")
                            try:
                                for audio_chunk in self.tts_model.stream_tts_sync(segment_to_process):
                                    yield audio_chunk
                                    
                                continue  # 一次只处理一个标点符号k
                            except Exception as e:
                                self.logger.error(f"Error during TTS stream: {e}", exc_info=True)
                                continue
            
            # 处理剩余的buffer
            if buffer_str.strip():
                # 如果末尾没有标点，添加一个句号
                if buffer_str[-1] not in ["。", "！", "？", "；", "，", "：", ".", "!", "?", ",", ";", ":"]:
                    buffer_str += "。"
                
                self.logger.debug(f"Processing final segment: '{buffer_str}'")
                for audio_chunk in self.tts_model.stream_tts_sync(buffer_str):
                    yield audio_chunk
                    
            self.logger.info("Finished TTS stream.")
            
        except Exception as e:
            self.logger.error(f"Error during echo processing: {e}", exc_info=True)
            return iter([])

def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication if script is re-run
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)

    root_logger.addHandler(stdout_handler)

    # logger.info("Logging configured test message.") # Keep or remove as needed

    llm_client = OpenAI(
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"), # Corrected typo: ollam -> ollama
        base_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/")
    )
    stt_model = get_stt_model()
    tts_model = get_tts_model(device="cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the handler
    # Configure ReplyOnStopWords with the handler's echo method and stop words
    echo_handler = EchoHandler(stt_model, tts_model, llm_client)
    echo_handler.stop_word_detected = lambda text: text.find("等一下") != -1

    stream = Stream(
        ReplyOnPause(
            fn=echo_handler.echo,
            model=None,
            can_interrupt=True,
        ),
        modality="audio",
        mode="send-receive",
    )

    logger.info("Starting Gradio UI.")
    # Consider adding debug=True for Gradio development/testing if helpful
    stream.ui.launch(share=True, server_name="0.0.0.0", server_port=7860) # share=True makes it accessible publicly

if __name__ == "__main__":
    main()
