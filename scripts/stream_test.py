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
    last_text = ""

    async with websockets.connect(args.url, max_size=None) as ws:
        cfg = json.loads(await ws.recv())
        print(f"config: {cfg}\n")

        async def reader():
            nonlocal last_text
            try:
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "ready_to_stop":
                        print(f"\n[{time.time()-started:6.2f}s] ready_to_stop")
                        return
                    if data.get("type") == "config":
                        continue
                    lines = " | ".join(
                        f"{l.get('text','').strip()}" for l in data.get("lines", [])
                    )
                    buf = data.get("buffer_transcription", "")
                    text = (lines + ("  ⟨" + buf + "⟩" if buf else "")).strip()
                    if text != last_text:
                        last_text = text
                        print(f"[{time.time()-started:6.2f}s] {text}")

            except websockets.ConnectionClosed:
                return

        task = asyncio.create_task(reader())
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i : i + step])
            if args.realtime > 0:
                await asyncio.sleep(min(CHUNK_SEC / args.realtime, 2))
        await ws.send(b"")
        try:
            await asyncio.wait_for(task, timeout=180)
        except asyncio.TimeoutError:
            print("timed out waiting for ready_to_stop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/asr")
    ap.add_argument("--realtime", type=float, default=1.0)
    ap.add_argument("--limit", type=float, default=0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
