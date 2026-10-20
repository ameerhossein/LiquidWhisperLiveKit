#!/usr/bin/env python3
"""Reference transcription with plain transformers (no CTranslate2).

Used to sanity-check that the model's own generation config (sot=51866,
<|fa|>=51908, <|transcribe|>=51968) produces the same text as the converted
faster-whisper path. Slow on CPU: run on a short clip.

Usage: source env.sh && .venv/bin/python scripts/hf_reference.py audio.wav [--seconds 30]
"""
import argparse
import time

import librosa
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

ROOT = "models/buzzasr-persian"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default=ROOT)
    ap.add_argument("--seconds", type=float, default=30)
    args = ap.parse_args()

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    audio = audio[: int(args.seconds * 16000)]
    print(f"audio: {len(audio)/16000:.2f}s")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).eval()
    gc = model.generation_config
    print(f"generation config: decoder_start={gc.decoder_start_token_id} "
          f"eos={gc.eos_token_id} lang={getattr(gc, 'language', None)} "
          f"task={getattr(gc, 'task', None)}")
    print(f"loaded in {time.time()-t0:.1f}s")

    feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    t1 = time.time()
    with torch.no_grad():
        ids = model.generate(feats, max_new_tokens=200)
    print(f"generate: {time.time()-t1:.1f}s, {ids.shape[1]} tokens")

    text_orig = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
    print(f"\nprocessor.tokenizer: {text_orig!r}")

    fixed = AutoProcessor.from_pretrained("models/buzzasr-persian-tok")
    print(f"fixed tokenizer:     {fixed.tokenizer.decode(ids[0], skip_special_tokens=True)!r}")
    print("\nfirst ids:", ids[0][:12].tolist())


if __name__ == "__main__":
    main()
