from main import EchoHandler


class FakeDelta:
    content = "回答。"
    role = function_call = tool_calls = None


class FakeChunk:
    choices = [type("Choice", (), {"delta": FakeDelta()})()]


class FakeSTT:
    def stt(self, audio):
        return "你好"


class FakeTTS:
    def stream_tts_sync(self, text):
        return iter(())


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter([FakeChunk()])


def test_echo_forwards_configured_request_options():
    completions = FakeCompletions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    handler = EchoHandler(
        FakeSTT(),
        FakeTTS(),
        client,
        "deepseek-v4-flash",
        {
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    )

    list(handler.echo((16000, b"audio"), []))

    assert completions.kwargs["model"] == "deepseek-v4-flash"
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["reasoning_effort"] == "high"
    assert completions.kwargs["extra_body"] == {
        "thinking": {"type": "enabled"}
    }
