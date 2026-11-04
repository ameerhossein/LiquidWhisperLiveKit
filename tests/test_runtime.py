import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import settings


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    import os
    for key in list(os.environ):
        if key.startswith("WLK_") or key in ("DEVICE", "MODEL_DIR"):
            monkeypatch.delenv(key)


def test_profile_defaults_and_explicit_baseline(monkeypatch):
    tuned, baseline = settings.decode_options(), settings.decode_options("baseline")
    assert tuned["beam_size"] == 1 and tuned["temperature"] == 0
    assert tuned["condition_on_previous_text"] is False
    assert baseline["beam_size"] == 5 and "temperature" not in baseline
    monkeypatch.setenv("WLK_TEMPERATURE", "")
    assert "temperature" not in settings.decode_options()


def test_device_is_explicit_and_cuda_cannot_fall_back(monkeypatch):
    assert settings.model_options("cpu")["device"] == "cpu"
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        settings.model_options("gpu")
    with pytest.raises(ValueError, match="Metal"):
        settings.device_name("metal")


def test_thread_controls_are_validated(monkeypatch):
    monkeypatch.setenv("WLK_CPU_THREADS", "6")
    monkeypatch.setenv("WLK_NUM_WORKERS", "2")
    assert settings.model_options()["cpu_threads"] == 6
    assert settings.model_options()["num_workers"] == 2
    monkeypatch.setenv("WLK_CPU_THREADS", "-1")
    with pytest.raises(ValueError):
        settings.model_options()


def test_environment_follows_relocated_checkout(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "env.sh").write_text((root / "env.sh").read_text())
    env = dict(os.environ)
    env.pop("PROJECT_ROOT", None)
    result = subprocess.run(["bash", "-c", 'source "$1/env.sh"; printf "%s" "$PROJECT_ROOT"',
                             "test", str(tmp_path)], env=env, capture_output=True, text=True, check=True)
    assert result.stdout == str(tmp_path)


def test_backend_device_profile_and_explicit_context_are_isolated(tmp_path, monkeypatch):
    import serve
    from whisperlivekit.session_asr_proxy import SessionASRProxy
    for name in settings.REQUIRED_MODEL_FILES:
        (tmp_path / name).write_text("test")
    model = Mock()
    model.transcribe.return_value = ([], None)
    factory = Mock(return_value=model)
    monkeypatch.setattr("faster_whisper.WhisperModel", factory)
    backend = serve._TunedFasterWhisperASR("fa", model_dir=str(tmp_path))
    assert factory.call_args.kwargs["device"] == "cpu"
    first = SessionASRProxy(backend, language="en", context="terminology")
    second = SessionASRProxy(backend)
    first.transcribe(np.zeros(1), init_prompt="rolling history")
    assert model.transcribe.call_args.kwargs["initial_prompt"] == "terminology"
    assert model.transcribe.call_args.kwargs["language"] == "en"
    second.transcribe(np.zeros(1), init_prompt="history")
    assert model.transcribe.call_args.kwargs["initial_prompt"] is None
    assert model.transcribe.call_args.kwargs["language"] == "fa"
    backend.use_prompt = True
    first.transcribe(np.zeros(1), init_prompt="history")
    assert model.transcribe.call_args.kwargs["initial_prompt"] == "terminology\nhistory"
    model.transcribe.side_effect = RuntimeError("test failure")
    with pytest.raises(RuntimeError):
        first.transcribe(np.zeros(1))
    assert backend.original_language == "fa"
    assert serve._SESSION_CONTEXT.get() is None


def test_factory_aliases_are_patched():
    import serve
    assert serve.whisper_online.FasterWhisperASR is serve._TunedFasterWhisperASR
    import whisperlivekit
    html = whisperlivekit.get_inline_ui_html()
    assert "unicode-bidi: plaintext" in html
    assert "class='textcontent' dir='auto'" in html


def test_pause_splitting_is_opt_in_and_other_commands_unchanged(monkeypatch):
    from serve import configure_server_args
    assert configure_server_args(["serve.py", "serve"])[-1] == "--no-vac"
    assert configure_server_args(["serve.py", "models"]) == ["serve.py", "models"]
    monkeypatch.setenv("WLK_VAC", "1")
    assert configure_server_args(["serve.py", "serve"]) == ["serve.py", "serve"]


@pytest.mark.parametrize("result", [{"text": ""}, {"text": "test", "words": []},
    {"text": "test", "words": [{"start": 0, "end": 0}]}])
def test_verification_rejects_empty_or_collapsed_results(result):
    from validate_result import validate
    with pytest.raises(ValueError):
        validate(result, words=True)


def test_verification_rejects_hallucinations_around_expected_sentence():
    from validate_result import validate
    with pytest.raises(ValueError):
        validate({"text": "hallucination expected text"}, expected="expected text")


class Messages:
    def __init__(self, messages):
        self.messages = messages

    def __aiter__(self):
        async def receive():
            for item in self.messages:
                yield json.dumps(item)
        return receive()


@pytest.mark.parametrize("messages", [[], [{"type": "error", "error": "failed"}],
    [{"status": "error", "error": "failed"}], [{"type": "ready_to_stop"}]])
def test_websocket_rejects_errors_missing_text_and_premature_close(messages):
    from stream_test import receive_results
    import time
    with pytest.raises(RuntimeError):
        asyncio.run(receive_results(Messages(messages), time.time(), require_text=True))


def test_websocket_requires_finalization_and_returns_committed_text():
    from stream_test import receive_results
    import time
    result = asyncio.run(receive_results(Messages([
        {"lines": [{"text": "سلام"}]}, {"type": "ready_to_stop"}
    ]), time.time(), require_text=True))
    assert result["text"] == "سلام"
    assert result["metrics"]["first_committed_s"] is not None


def test_websocket_finish_timeout_is_an_error(monkeypatch):
    import stream_test

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def recv(self):
            return json.dumps({"type": "config", "useAudioWorklet": True})

        async def send(self, data):
            pass

        def __aiter__(self):
            async def wait_forever():
                await asyncio.Future()
                yield "unreachable"
            return wait_forever()

    monkeypatch.setattr(stream_test.websockets, "connect", lambda *a, **k: Socket())
    monkeypatch.setattr(stream_test.sf, "read", lambda *a, **k: (np.zeros((1600, 1)), 16000))
    with pytest.raises(TimeoutError):
        asyncio.run(stream_test.run(SimpleNamespace(audio="mock.wav", limit=0, realtime=0,
            url="ws://unused", timeout=.01, require_text=True, json_output=None)))
