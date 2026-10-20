#!/usr/bin/env python3
"""Benchmark the converted model on CPU: threads x beam x audio length."""
import argparse
import time

import librosa
from faster_whisper import WhisperModel

ROOT = "models/buzzasr-persian-ct2-int8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", default="testdata/persian-sample.wav", nargs="?")
    ap.add_argument("--audio-len", type=float, default=0, help="truncate audio to N seconds")
    args = ap.parse_args()

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    if args.audio_len:
        audio = audio[: int(args.audio_len * 16000)]
    print(f"audio {len(audio)/16000:.2f}s\n")

    for threads in (4, 8, 12):
        for beam in (1, 5):
            m = WhisperModel(ROOT, device="cpu", compute_type="int8", cpu_threads=threads)
            t0 = time.time()
            segs, info = m.transcribe(audio, language="fa", beam_size=beam,
                                      condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segs)
            dt = time.time() - t0
            print(f"threads={threads:2d} beam={beam} -> {dt:6.2f}s  "
                  f"RTF={dt/info.duration:5.2f}  {text[:70]!r}")
            del m


if __name__ == "__main__":
    main()
