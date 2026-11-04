#!/usr/bin/env python3
"""Pairwise ECAPA speaker similarity for the fa_clips speakers.

Diarization scores only mean something if the speakers are acoustically
distinct; this checks that with an independent embedding model (SpeechBrain
ECAPA-VoxCeleb). Run with the voiceprint venv, which already has speechbrain:

  ~/Hobbies/voiceprint/.venv/bin/python scripts/speaker_similarity.py
"""
import itertools
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

CLIPS = Path("/home/ameer/Hobbies/voiceprint/fa_clips")


def embed(model, audio):
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(wav).squeeze().numpy()
    return emb / np.linalg.norm(emb)


def main():
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=".cache/sb-ecapa"
    )
    speakers = sorted(p.name for p in CLIPS.iterdir() if p.is_dir())
    embs = {}
    for spk in speakers:
        vecs = []
        for f in sorted((CLIPS / spk).glob("*.wav"))[:4]:
            audio, sr = sf.read(str(f), dtype="float32", always_2d=True)
            assert sr == 16000
            vecs.append(embed(model, audio.mean(axis=1)))
        embs[spk] = np.mean(vecs, axis=0) / np.linalg.norm(np.mean(vecs, axis=0))
        print(f"{spk}: {len(vecs)} clips")

    print("\ncosine similarity between speakers:")
    print("             " + "".join(f"{s[-5:]:>8s}" for s in speakers))
    for a in speakers:
        row = "".join(f"{np.dot(embs[a], embs[b]):8.2f}" for b in speakers)
        print(f"{a:>12s} {row}")

    print("\nmost similar pairs:")
    pairs = sorted(
        ((np.dot(embs[a], embs[b]), a, b) for a, b in itertools.combinations(speakers, 2)),
        reverse=True,
    )
    for sim, a, b in pairs[:4]:
        print(f"  {a} <-> {b}: {sim:.2f}")


if __name__ == "__main__":
    main()
