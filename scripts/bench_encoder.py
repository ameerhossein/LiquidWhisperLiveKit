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
from settings import decode_options, model_options, model_path, print_settings, require_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--threads", type=int, default=0, help="0 = CTranslate2 default")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--profile", choices=("tuned", "baseline"), default=None)
    args = ap.parse_args()

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    print(f"audio {len(audio)/16000:.2f}s threads={args.threads or 'default'} beam={args.beam}")

    options = decode_options(args.profile, args.beam)
    runtime = model_options("cpu", args.threads)
    print_settings(options, runtime)
    m = WhisperModel(str(require_model(args.model or model_path("cpu"))), **runtime)
    feats = m.feature_extractor(audio)
    # encoder only (padded to 30 s window)
    # Match the first 30-second window used by transcribe().
    feats = np.pad(feats[:, :3000], ((0, 0), (0, max(0, 3000 - feats.shape[-1]))))
    enc_in = feats[None, :]
    t0 = time.time()
    m.model.encode(enc_in)
    enc = time.time() - t0

    t0 = time.time()
    segs, info = m.transcribe(audio, language="fa", word_timestamps=True, **options)
    text = " ".join(s.text.strip() for s in segs)
    total = time.time() - t0
    print(f"encoder(30s window): {enc:6.2f}s")
    print(f"full transcribe:     {total:6.2f}s  (remaining work≈{total-enc:.2f}s)  "
          f"RTF={total/info.duration:5.2f}")
    print(f"text: {text[:120]!r}")


if __name__ == "__main__":
    main()
