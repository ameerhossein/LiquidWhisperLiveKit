#!/usr/bin/env python3
"""Transcribe a file with the converted BuzzASR/persian CT2 model.

Usage: source env.sh && .venv/bin/python scripts/transcribe.py audio.wav [--words]
"""
import argparse
import time
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default=str(ROOT / "models/buzzasr-persian-ct2-int8"))
    ap.add_argument("--words", action="store_true")
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    t0 = time.time()
    segments, info = model.transcribe(
        args.audio, language="fa", beam_size=args.beam, word_timestamps=args.words
    )
    segments = list(segments)
    elapsed = time.time() - t0
    for seg in segments:
        print(f"[{seg.start:6.2f} -> {seg.end:6.2f}] {seg.text.strip()}")
        if args.words:
            print("   ", " ".join(f"{w.word.strip()}({w.start:.2f})" for w in seg.words or []))
    print(f"\nfaster-whisper: {elapsed:.2f}s for {info.duration:.2f}s audio "
          f"(RTF {elapsed / max(info.duration, 1e-6):.2f})")


if __name__ == "__main__":
    main()
