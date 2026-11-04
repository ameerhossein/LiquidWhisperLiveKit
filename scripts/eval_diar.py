#!/usr/bin/env python3
"""Score WhisperLiveKit's streaming Sortformer diarization against ground truth.

Runs the same objects the server uses (SortformerDiarization + the online
session wrapper), feeds a WAV in small chunks, and compares the emitted
SpeakerSegments with the turn boundaries in <audio>.json.

Metrics:
  * frame-level DER (miss / false alarm / confusion) with an optimal
    hypothesis->reference speaker mapping (Hungarian),
  * per-turn purity: what share of each true turn got one predicted speaker,
  * predicted vs true speaker count.

Usage: source env.sh && .venv/bin/python scripts/eval_diar.py testdata/diar-4spk.wav
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

from diar_stream import stream_diarization

HOP = 0.01  # 10 ms frames


def labels_from_turns(turns, n_frames):
    lab = np.zeros(n_frames, dtype=np.int16)  # 0 = silence
    names = {}
    for t in turns:
        sid = names.setdefault(t["speaker"], len(names) + 1)
        a = int(round(t["start"] / HOP))
        b = int(round(t["end"] / HOP))
        lab[max(a, 0):min(b, n_frames)] = sid
    return lab, names


def labels_from_segments(segments, n_frames):
    lab = np.zeros(n_frames, dtype=np.int16)
    ids = {}
    for s in segments:
        sid = ids.setdefault(int(s.speaker), len(ids) + 1)
        a = int(round(s.start / HOP))
        b = int(round(s.end / HOP))
        lab[max(a, 0):min(b, n_frames)] = sid
    return lab, ids


def der(ref, hyp, n_ref_speakers, n_hyp_speakers):
    """Frame-level DER with optimal one-to-one speaker mapping."""
    ref_speech = ref > 0
    hyp_speech = hyp > 0
    total = ref_speech.sum()

    # Confusion between simultaneously active labelled frames.
    both = ref_speech & hyp_speech
    conf = np.zeros((n_ref_speakers + 1, n_hyp_speakers + 1), dtype=np.int64)
    for r, h in zip(ref[both], hyp[both]):
        conf[r, h] += 1

    # Map each hyp speaker to at most one ref speaker (dic region only).
    sub = conf[1:, 1:]
    rows, cols = linear_sum_assignment(-sub)
    mapping = {c + 1: r + 1 for r, c in zip(rows, cols)}

    mapped = np.zeros_like(hyp)
    for h, r in mapping.items():
        mapped[hyp == h] = r
    correct = ((mapped == ref) & both).sum()
    confusion = both.sum() - correct
    false_alarm = (hyp_speech & ~ref_speech).sum()
    missed = (ref_speech & ~hyp_speech).sum()

    return {
        "der": (missed + false_alarm + confusion) / max(total, 1),
        "missed": missed / max(total, 1),
        "false_alarm": false_alarm / max(total, 1),
        "confusion": confusion / max(total, 1),
        "mapping": mapping,
        "ref_speech_s": total * HOP,
        "hyp_speech_s": hyp_speech.sum() * HOP,
    }


async def run(audio_path: Path, chunk=0.5):
    from whisperlivekit.diarization.sortformer_backend import (
        SortformerDiarization, SortformerDiarizationOnline,
    )
    turns = json.loads(audio_path.with_suffix(".json").read_text())
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError("Expected 16 kHz audio")

    print("loading Sortformer...")
    shared = SortformerDiarization()
    online = SortformerDiarizationOnline(shared_model=shared)
    print(f"device={shared.diar_model.device} max_speakers={online.max_speakers} "
          f"chunk={online.chunk_duration_seconds:.2f}s")

    import time
    t0 = time.time()
    segments = await stream_diarization(online, audio, sr, chunk)
    elapsed = time.time() - t0

    n_frames = int(round(len(audio) / sr / HOP))
    ref, ref_names = labels_from_turns(turns, n_frames)
    hyp, hyp_ids = labels_from_segments(segments, n_frames)

    res = der(ref, hyp, len(ref_names), len(hyp_ids))
    print("DER policy: 10 ms frames, no collar, single-speaker labels, silence scored; overlap unsupported")
    print(f"\nstreamed {len(audio)/sr:.1f}s in {elapsed:.1f}s "
          f"(RTF {elapsed/(len(audio)/sr):.2f}), {len(segments)} segments")
    print(f"true speakers: {len(ref_names)}  predicted: {len(hyp_ids)}  "
          f"mapping hyp->ref: {res['mapping']}")
    print(f"DER {res['der']*100:5.1f}%  "
          f"(miss {res['missed']*100:.1f}%  "
          f"false-alarm {res['false_alarm']*100:.1f}%  "
          f"confusion {res['confusion']*100:.1f}%)")

    # Per-turn purity: fraction of the turn assigned to its dominant speaker.
    print("\nturn purity:")
    ok = 0
    for t in turns:
        a, b = int(t["start"] / HOP), int(t["end"] / HOP)
        seg = hyp[max(a, 0):min(b, n_frames)]
        vals, counts = np.unique(seg[seg > 0], return_counts=True)
        if len(vals) == 0:
            print(f"  {t['start']:6.2f}-{t['end']:6.2f} {t['speaker']:10s} -> unmapped")
            continue
        dom = int(vals[np.argmax(counts)])
        purity = counts.max() / len(seg)
        ok += purity > 0.8
        print(f"  {t['start']:6.2f}-{t['end']:6.2f} {t['speaker']:10s} -> "
              f"spk{dom} purity {purity*100:5.1f}%")
    print(f"\nturns with >=80% single-speaker coverage: {ok}/{len(turns)}")

    out = audio_path.with_name(audio_path.stem + "-diar.json")
    out.write_text(json.dumps(
        [{"start": s.start, "end": s.end, "speaker": int(s.speaker)} for s in segments],
        indent=1))
    print("segments written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", default="testdata/diar-4spk.wav")
    ap.add_argument("--chunk", type=float, default=0.5)
    args = ap.parse_args()
    asyncio.run(run(Path(args.audio), args.chunk))
