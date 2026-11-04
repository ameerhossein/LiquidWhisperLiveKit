#!/usr/bin/env python3
"""Transcribe a file with the converted BuzzASR/persian CT2 model.

Usage: source env.sh && .venv/bin/python scripts/transcribe.py audio.wav [--words]
Device follows $DEVICE (cpu|gpu); --device overrides it.
"""
import argparse
import os
import time
from pathlib import Path

from faster_whisper import WhisperModel
from settings import decode_options, model_options, model_path, print_settings, require_model

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default=None,
                    help="CTranslate2 directory (default: int8 for cpu, fp16 for gpu)")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    ap.add_argument("--words", action="store_true")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--profile", choices=("tuned", "baseline"), default=None)
    ap.add_argument("--json-output", help="Write text and word timings for verification")
    args = ap.parse_args()

    model_dir = str(require_model(args.model or model_path(args.device)))
    runtime = model_options(args.device)
    options = decode_options(args.profile, args.beam)
    print_settings(options, runtime)
    model = WhisperModel(model_dir, **runtime)
    t0 = time.monotonic()
    segments, info = model.transcribe(
        args.audio, language="fa", word_timestamps=args.words, **options
    )
    segments = list(segments)
    elapsed = time.monotonic() - t0
    if not any(seg.text.strip() for seg in segments):
        raise RuntimeError("No speech transcript produced")
    if args.json_output:
        import json
        import resource
        import sys
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
        Path(args.json_output).write_text(json.dumps({
            "text": " ".join(s.text.strip() for s in segments),
            "words": [dict(word=w.word, start=w.start, end=w.end)
                      for s in segments for w in (s.words or [])],
            "duration": info.duration,
            "metrics": {"asr_s": elapsed, "rtf": elapsed / max(info.duration, 1e-6),
                        "peak_process_rss_mb": peak_mb},
        }, ensure_ascii=False), encoding="utf-8")
    for seg in segments:
        print(f"[{seg.start:6.2f} -> {seg.end:6.2f}] {seg.text.strip()}")
        if args.words:
            print("   ", " ".join(f"{w.word.strip()}({w.start:.2f})" for w in seg.words or []))
    print(f"\nfaster-whisper: {elapsed:.2f}s for {info.duration:.2f}s audio "
          f"(RTF {elapsed / max(info.duration, 1e-6):.2f})")


if __name__ == "__main__":
    main()
