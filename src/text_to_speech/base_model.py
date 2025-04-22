import numpy as np
from numpy.typing import NDArray
from typing import Optional, Protocol
from collections.abc import AsyncGenerator, Generator

class TTSOptions:
    """
    Options for Text-to-Speech synthesis.
    """
    time_step: int = 32
    p_w: float = 1.6
    t_w: float = 2.5
    lang: Optional[str] = None

    def __init__(self):
        pass

class TTSModel(Protocol):
    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]: 
        pass

    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]: 
        pass
    
    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        pass
