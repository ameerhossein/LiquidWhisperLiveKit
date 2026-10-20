#!/usr/bin/env python3
"""Measure encoder / decoder cost separately for the converted model.

Usage: python bench_encoder.py audio.wav [--threads N] [--beam N]
Runs in a fresh process so thread-pool and thermal state do not leak between
configurations.
"""
import argparse
import time

import librosa
import numpy as np
from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--threads", type=int, default=0, help="0 = CTranslate2 default")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--model", default="models/buzzasr-persian-ct2-int8")
    args = ap.parse_args()

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    print(f"audio {len(audio)/16000:.2f}s threads={args.threads or 'default'} beam={args.beam}")

    m = WhisperModel(args.model, device="cpu", compute_type="int8",
                     cpu_threads=args.threads)
    feats = m.feature_extractor(audio)
    # encoder only (padded to 30 s window)
    enc_in = feats[None, :]
    t0 = time.time()
    m.model.encode(enc_in)
    enc = time.time() - t0

    t0 = time.time()
    segs, info = m.transcribe(audio, language="fa", beam_size=args.beam,
                              condition_on_previous_text=False,
                              without_timestamps=False)
    text = " ".join(s.text.strip() for s in segs)
    total = time.time() - t0
    print(f"encoder(30s window): {enc:6.2f}s")
    print(f"full transcribe:     {total:6.2f}s  (decoder≈{total-enc:.2f}s)  "
          f"RTF={total/info.duration:5.2f}")
    print(f"text: {text[:120]!r}")


if __name__ == "__main__":
    main()
