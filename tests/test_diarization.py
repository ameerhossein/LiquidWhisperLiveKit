import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diar_stream import stream_diarization
from whisperlivekit.timed_objects import SpeakerSegment


def test_canonical_keeps_history_and_partial_tail():
    from compare_diar_drivers import canonical

    class Modules:
        n_spk = 2
        chunk_len = 12.5
        subsampling_factor = 8

        def init_streaming_state(self, **kwargs):
            return None

    class Pre:
        _cfg = SimpleNamespace(window_stride=.01)

        def get_features(self, chunk, length):
            assert chunk.device.type == "cpu" and chunk.shape[-1] == 16000
            return torch.zeros((1, 128, 100)), torch.tensor([100])

    class Model:
        device = "cpu"
        preprocessor = Pre()
        sortformer_modules = Modules()

        def forward_streaming_step(self, **kwargs):
            history = kwargs["total_preds"]
            return None, torch.cat([history, torch.tensor([[[.9, .1]] * 10])], dim=1)

    segments = canonical(SimpleNamespace(diar_model=Model()), np.zeros(52000, dtype=np.float32))
    assert len(segments) == 4
    assert segments[-1].end == 3.25
    assert sum(s.end - s.start for s in segments) == 3.25


@pytest.mark.parametrize("chunk", [.25, 1, 3])
def test_drain_handles_large_chunks_and_clips_padding(chunk):
    class Online:
        chunk_duration_seconds = 1
        buffer_audio = np.zeros(0)
        time = 0

        def insert_audio_chunk(self, audio):
            self.buffer_audio = np.concatenate([self.buffer_audio, audio])

        async def diarize(self):
            self.buffer_audio = self.buffer_audio[16000:]
            self.time += 1
            return [SpeakerSegment(start=self.time - 1, end=self.time, speaker=1)]

    online = Online()
    segments = asyncio.run(stream_diarization(online, np.zeros(52000), chunk=chunk))
    assert len(online.buffer_audio) == 0
    assert len(segments) == 4
    assert segments[-1].end == 3.25


def test_invalid_chunk_rejected():
    with pytest.raises(ValueError):
        asyncio.run(stream_diarization(None, [], chunk=0))
