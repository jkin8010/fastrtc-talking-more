import json
import os
import logging
import sys
import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import torch
import gradio as gr
import numpy as np
from fastrtc import (AdditionalOutputs, ReplyOnPause, Stream)
import uvicorn
from pause_detection.fsmn.model import get_fsmn_vad_model
from text_to_speech.megatts.model import get_tts_model, TTSModel
from speech_to_text.funasr.model import get_stt_model, STTModel
from openai import OpenAI
from logging import getLogger
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

logger = getLogger(__name__)

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "public")

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


class Message(BaseModel):
    role: str
    content: str


class InputData(BaseModel):
    webrtc_id: str
    chatbot: List[Message]


def clean_text_for_tts(text):
    """
    清理文本中的Markdown特殊字符和其他可能导致TTS问题的内容
    """
    # 移除Markdown标题符号
    text = re.sub(r'^#+\s+', '', text)
    text = re.sub(r'\n#+\s+', '\n', text)
    
    # 移除Markdown格式符号
    text = re.sub(r'\*\*|\*|__|\\_', '', text)
    
    # 移除代码块标记
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 移除URL
    text = re.sub(r'https?://\S+', '', text)
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class EchoHandler:
    def __init__(self, stt_model, tts_model, llm_client):
        self.stt_model: STTModel = stt_model
        self.tts_model: TTSModel = tts_model
        self.llm_client: OpenAI = llm_client
        self.logger = getLogger(__name__ + ".EchoHandler")

    def echo(self, audio, chatbot: Optional[List[Dict]] = None):
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

            # 更新 chatbot 历史记录
            if chatbot is None:
                chatbot = []
            chatbot.append({"role": "user", "content": re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', prompt)})
            yield AdditionalOutputs(chatbot)
            
            # 准备 LLM 消息
            messages = [{"role": d["role"], "content": re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', d["content"])} if isinstance(d, dict) else d for d in chatbot]
            
            response_stream = self.llm_client.chat.completions.create(
                model="qwen2.5",
                messages=messages,
                stream=True
            )

            # Process LLM stream and collect full response
            buffer_str = ""
            punctuation_marks = ["。", "！", "？", "；", ".", "!", "?", ";", "\n"]
            
            self.logger.info("Starting to receive LLM stream.")
            print("LLM response stream: ", end="", flush=True)
            
            # 标记是否接收到结束信号
            received_end = False
            # 记录连续空内容的次数
            empty_content_count = 0
            
            for chunk in response_stream:
                delta = chunk.choices[0].delta
                
                # 检查是否是空内容，可能表示流结束
                if delta.content is None and delta.role is None and delta.function_call is None and delta.tool_calls is None:
                    empty_content_count += 1
                    # 如果连续多次收到空内容，认为是流结束
                    if empty_content_count >= 3:
                        received_end = True
                        self.logger.info("Detected end of stream after multiple empty deltas")
                    # 处理缓冲区内容，避免因空内容而中断
                    if buffer_str and empty_content_count == 2:
                        self.logger.info(f"Processing buffer on empty content: '{buffer_str}'")
                        if not any(mark in buffer_str for mark in punctuation_marks):
                            # 如果没有标点，尝试添加一个句号处理
                            temp_buffer = buffer_str + "。"
                            try:
                                temp_buffer = clean_text_for_tts(temp_buffer)
                                for audio_chunk in self.tts_model.stream_tts_sync(temp_buffer):
                                    yield audio_chunk
                                buffer_str = ""  # 清空缓冲区，避免重复处理
                            except Exception as e:
                                self.logger.error(f"Error processing buffer on empty content: {e}")
                    continue
                
                # 重置空内容计数器    
                empty_content_count = 0
                
                if delta and delta.content:
                    text_chunk = delta.content
                    print(text_chunk, end="", flush=True)  # 打印即时反馈
                    buffer_str += text_chunk
                    
                    # 检查每个标点符号是否在新增内容中出现
                    for mark in punctuation_marks:
                        if mark in text_chunk:
                            # 分离出要处理的部分
                            parts = buffer_str.split(mark)
                            # 处理除最后一部分外的所有部分
                            for i in range(len(parts) - 1):
                                segment = parts[i]
                                if segment:  # 确保部分不为空
                                    segment_to_process = segment + mark
                                    # 更新 chatbot 历史记录
                                    chatbot.append({"role": "assistant", "content": re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', segment_to_process)})
                                
                                    yield AdditionalOutputs(chatbot)
                                    # 清理文本中的Markdown特殊字符
                                    segment_to_process = clean_text_for_tts(segment_to_process)
                                    self.logger.debug(f"Processing segment during streaming: '{segment_to_process}'")
                                    try:
                                        for audio_chunk in self.tts_model.stream_tts_sync(segment_to_process):
                                            yield audio_chunk
                                    except Exception as e:
                                        self.logger.error(f"Error during TTS stream: {e}", exc_info=True)
                                        # 发生错误时继续处理，不中断整个流程
                            
                            # 保留最后一部分到buffer中
                            buffer_str = parts[-1]
                            break  # 处理完一个标点符号的所有实例后退出循环
            
            # 检查是否收到结束信号且有内容需要处理
            self.logger.info(f"Stream ended, remaining buffer: '{buffer_str}'")
            
            # 处理剩余的buffer
            if buffer_str.strip():
                # 如果末尾没有标点，添加一个句号
                if not any(buffer_str.endswith(mark) for mark in punctuation_marks):
                    buffer_str += "。"
               
                # 更新 chatbot 历史记录
                chatbot.append({"role": "assistant", "content": re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', buffer_str)})
              
                yield AdditionalOutputs(chatbot) 
                # 清理文本中的Markdown特殊字符
                buffer_str = clean_text_for_tts(buffer_str)
                self.logger.debug(f"Processing final segment: '{buffer_str}'")
                
                try:
                    for audio_chunk in self.tts_model.stream_tts_sync(buffer_str):
                        yield audio_chunk
                except Exception as e:
                    self.logger.error(f"Error processing final segment in TTS: {e}", exc_info=True)
                    # 发生错误时继续，不中断流程
            
            self.logger.info("Finished TTS stream.")  
                
        except Exception as e:
            self.logger.error(f"Error during echo processing: {e}", exc_info=True)
            return iter([])

def main():
    # Configure logging
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
    
    # 设置第三方库的日志级别，避免过多噪音
    logging.getLogger('aioice').setLevel(logging.INFO)
    logging.getLogger('aiohttp').setLevel(logging.INFO)
    logging.getLogger('modelscope').setLevel(logging.INFO)
    logging.getLogger('funasr').setLevel(logging.INFO)

    # logger.info("Logging configured test message.") # Keep or remove as needed

    llm_client = OpenAI(
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"), # Corrected typo: ollam -> ollama
        base_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/"),
    )
    # fsmn_vad_model = get_fsmn_vad_model()
    stt_model = get_stt_model()
    tts_model = get_tts_model(device="cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the handler
    # Configure ReplyOnStopWords with the handler's echo method and stop words
    echo_handler = EchoHandler(stt_model, tts_model, llm_client)
    echo_handler.stop_word_detected = lambda text: text.find("等一下") != -1

    chatbot = gr.Chatbot(type="messages")
    stream = Stream(
        ReplyOnPause(
            fn=echo_handler.echo,
            can_interrupt=True,
        ),
        modality="audio",
        mode="send-receive",
        additional_outputs_handler=lambda a, b: b,
        additional_inputs=[chatbot],
        additional_outputs=[chatbot],
    )

    logger.info("Starting Gradio UI.")
    logger.info(f"Gradio UI API: {stream.ui.get_api_info(all_endpoints=True)}")

    # Consider adding debug=True for Gradio development/testing if helpful
    app = FastAPI()
    stream.mount(app)
        
    @app.get("/")
    async def _():
        html_content = open(os.path.join(PUBLIC_DIR, "index.html"), "r").read()
        html_content = html_content.replace("__RTC_CONFIGURATION__", "{}")
        return HTMLResponse(content=html_content, status_code=200)


    @app.post("/input_hook")
    async def _(body: InputData):
        stream.set_input(body.webrtc_id, body.chatbot)
        return {"status": "ok"}


    @app.get("/outputs")
    def _(webrtc_id: str):
        async def output_stream():
            async for output in stream.output_stream(webrtc_id):
                chatbot = output.args[0]
                yield f"event: output\ndata: {json.dumps(chatbot[-1])}\n\n"

        return StreamingResponse(output_stream(), media_type="text/event-stream")
   
    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860, server_name="0.0.0.0")
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()