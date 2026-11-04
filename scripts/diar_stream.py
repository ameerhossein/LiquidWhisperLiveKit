"""Chunk-complete driving and tail clipping for streaming diarization evals."""
import math
import numpy as np


def clip_segments(segments, duration):
    result = []
    for segment in segments:
        start, end = max(0.0, segment.start), min(duration, segment.end)
        if end > start:
            # Preserve the runtime's own segment type and speaker convention.
            from dataclasses import replace
            result.append(replace(segment, start=start, end=end))
    return result


async def stream_diarization(online, audio, sr=16000, chunk=0.5):
    if not math.isfinite(chunk) or chunk <= 0 or int(chunk * sr) < 1:
        raise ValueError("chunk must contain at least one sample")
    if sr != 16000:
        raise ValueError("Diarization input must be 16 kHz")
    threshold = int(round(online.chunk_duration_seconds * sr))
    if threshold < 1:
        raise ValueError("Model chunk duration must be positive")
    segments = []

    async def drain():
        while len(online.buffer_audio) >= threshold:
            before = len(online.buffer_audio)
            segments.extend(await online.diarize())
            if len(online.buffer_audio) >= before:
                raise RuntimeError("Diarizer did not consume a complete chunk")

    step = int(chunk * sr)
    for i in range(0, len(audio), step):
        online.insert_audio_chunk(audio[i:i + step])
        await drain()
    if len(online.buffer_audio):
        online.insert_audio_chunk(np.zeros(threshold - len(online.buffer_audio), dtype=np.float32))
        await drain()
    return clip_segments(segments, len(audio) / sr)
