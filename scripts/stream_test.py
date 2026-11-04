#!/usr/bin/env python3
"""Stream a WAV file into WhisperLiveKit's /asr WebSocket and print updates.

Usage:
  source env.sh
  .venv/bin/python scripts/stream_test.py audio.wav [--url ws://127.0.0.1:8000/asr]
                                                 [--realtime 1.0] [--limit 30]

Sends raw s16le 16 kHz mono PCM, so start the server with `--pcm-input`.
--realtime   playback speed factor (1.0 = real time, 0 = as fast as possible)
"""
import argparse
import asyncio
import json
import math
import time

import numpy as np
import soundfile as sf
import websockets

SAMPLE_RATE = 16000
CHUNK_SEC = 0.25


async def run(args):
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if args.limit:
        audio = audio[: int(args.limit * SAMPLE_RATE)]
    duration = len(audio) / SAMPLE_RATE
    print(f"audio: {duration:.2f}s  sr={SAMPLE_RATE}  url={args.url}")

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    step = int(CHUNK_SEC * SAMPLE_RATE) * 2
    started = time.time()
    async with websockets.connect(args.url, max_size=16 * 1024 * 1024) as ws:
        cfg = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
        if cfg.get("type") != "config" or not cfg.get("useAudioWorklet"):
            raise RuntimeError(f"Server did not accept PCM input: {cfg}")
        print(f"config: {cfg}\n")
        task = asyncio.create_task(receive_results(ws, started, args.require_text))
        try:
            for i in range(0, len(pcm), step):
                if task.done():
                    await task
                    raise RuntimeError("Server finalized before end of input")
                await ws.send(pcm[i:i + step])
                if args.realtime > 0:
                    await asyncio.sleep(len(pcm[i:i + step]) / (2 * SAMPLE_RATE * args.realtime))
            await ws.send(b"")
            result = await asyncio.wait_for(task, timeout=args.timeout)
            result["duration"] = duration
            if args.json_output:
                from pathlib import Path
                Path(args.json_output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return result
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def receive_results(ws, started, require_text=False):
    latest = {}
    first_partial = first_committed = None
    updates = 0
    last_rendered = None
    async for message in ws:
        data = json.loads(message)
        elapsed = time.time() - started
        if data.get("type") == "error" or data.get("status") == "error" or data.get("error"):
            raise RuntimeError(data.get("error") or "Server transcription error")
        if data.get("type") == "ready_to_stop":
            text = " ".join(line.get("text", "").strip() for line in latest.get("lines", [])).strip()
            if require_text and not text:
                raise RuntimeError("Server finalized without committed text")
            print(f"[{elapsed:6.2f}s] ready_to_stop")
            return dict(text=text, lines=latest.get("lines", []), metrics=dict(
                first_partial_s=first_partial, first_committed_s=first_committed,
                final_s=elapsed, updates=updates))
        if data.get("type") == "config":
            continue
        latest = data
        updates += 1
        committed = " ".join(line.get("text", "").strip() for line in data.get("lines", []))
        partial = data.get("buffer_transcription", "")
        if (committed.strip() or partial.strip()) and first_partial is None:
            first_partial = elapsed
        if committed.strip() and first_committed is None:
            first_committed = elapsed
        rendered = f"{committed} {'⟨' + partial + '⟩' if partial else ''}".strip()
        if rendered and rendered != last_rendered:
            print(f"[{elapsed:6.2f}s] {rendered}")
            last_rendered = rendered
    raise RuntimeError("WebSocket closed before ready_to_stop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/asr")
    ap.add_argument("--realtime", type=float, default=1.0)
    ap.add_argument("--limit", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=900, help="Handshake and finalization timeout")
    ap.add_argument("--require-text", action="store_true")
    ap.add_argument("--json-output")
    args = ap.parse_args()
    if (not all(math.isfinite(v) for v in (args.timeout, args.realtime, args.limit))
            or args.timeout <= 0 or args.realtime < 0 or args.limit < 0):
        ap.error("timeout must be positive; realtime and limit must be nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
