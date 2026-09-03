from openai import OpenAI

from .config import LLMConfig


def create_llm_client(config: LLMConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)
