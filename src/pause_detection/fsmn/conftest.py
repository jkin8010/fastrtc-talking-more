from pathlib import Path

import pytest
from pause_detection.fsmn.model import get_fsmn_vad_model


@pytest.fixture
def input_file() -> str:
    """Use the repository's checked-in audio sample for VAD tests."""
    audio_file = Path(__file__).resolve().parents[2] / "asr_example.wav"
    if not audio_file.is_file():
        pytest.fail(f"Required test audio file is missing: {audio_file}")
    return str(audio_file)


@pytest.fixture
def fsmn_vad_model():
    """Load the shared FSMN VAD model required by the pause test."""
    return get_fsmn_vad_model()
