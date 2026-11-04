#!/usr/bin/env python3
"""Compare two ways of driving NeMo streaming Sortformer, same audio/model.

  canonical : one 1 s chunk -> features -> forward_streaming_step(offsets 0)
              (this is what NeMo's own streaming loop does)
  wlk       : WhisperLiveKit's SortformerDiarizationOnline (prepends 99 mel
              frames of context and passes left_offset=8/right_offset=8)

Both paths are scored with scripts/eval_diar.py's DER code.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_diar import der, labels_from_segments, labels_from_turns  # noqa: E402

from diar_stream import stream_diarization, clip_segments
from whisperlivekit.timed_objects import SpeakerSegment  # noqa: E402


def canonical(shared, audio, sr=16000):
    """NeMo-style loop: raw 1 s chunks, no manual context, zero offsets."""
    if sr != 16000:
        raise ValueError("Expected 16 kHz audio")
    model = shared.diar_model
    pre = shared.diar_model.preprocessor
    spk = int(model.sortformer_modules.n_spk)
    # NeMo's own synchronous streaming state (empty cache, grown by the model)
    state = model.sortformer_modules.init_streaming_state(
        batch_size=1, async_streaming=False, device=model.device)
    total = torch.zeros((1, 0, spk), device=model.device)
    segments = []
    prev_frames = 0
    chunk_len = int(model.sortformer_modules.chunk_len *
                    model.sortformer_modules.subsampling_factor *
                    pre._cfg.window_stride * sr)  # samples per chunk
    chunk_len = max(chunk_len, sr // 2)
    for i in range(0, len(audio), chunk_len):
        samples = audio[i:i + chunk_len]
        samples = np.pad(samples, (0, chunk_len - len(samples)))
        chunk = torch.tensor(samples, dtype=torch.float32, device=model.device).unsqueeze(0)
        length = torch.tensor([chunk.shape[1]], device=model.device)
        with torch.inference_mode():
            feats, feat_len = pre.get_features(chunk, length)
            feats = feats.transpose(1, 2)
            state, total_preds = model.forward_streaming_step(
                processed_signal=feats, processed_signal_length=feat_len,
                streaming_state=state, total_preds=total)
        total = total_preds
        new = total_preds[0][prev_frames:]
        prev_frames = total_preds.shape[1]
        if len(new) == 0:
            continue
        labels = new.argmax(dim=1).tolist()
        frame_dt = chunk_len / sr / len(labels)
        base = i / sr
        cur = labels[0]
        start = base
        for j, lab in enumerate(labels):
            if lab != cur:
                segments.append(SpeakerSegment(speaker=cur, start=start,
                                               end=base + j * frame_dt))
                cur, start = lab, base + j * frame_dt
        segments.append(SpeakerSegment(speaker=cur, start=start,
                                       end=base + len(labels) * frame_dt))
    return clip_segments(segments, len(audio) / sr)


async def wlk(shared, audio, sr=16000, chunk=0.5):
    from whisperlivekit.diarization.sortformer_backend import SortformerDiarizationOnline
    online = SortformerDiarizationOnline(shared_model=shared)
    return await stream_diarization(online, audio, sr, chunk)


def score(name, segments, turns, duration):
    n_frames = int(round(duration / 0.01))
    ref, ref_names = labels_from_turns(turns, n_frames)
    hyp, hyp_ids = labels_from_segments(segments, n_frames)
    res = der(ref, hyp, len(ref_names), len(hyp_ids))
    print(f"{name:10s} segments={len(segments):4d} speakers={len(hyp_ids)} "
          f"DER={res['der']*100:5.1f}%  (miss {res['missed']*100:4.1f} "
          f"fa {res['false_alarm']*100:4.1f} conf {res['confusion']*100:4.1f})")
    return res["der"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    args = ap.parse_args()
    path = Path(args.audio)
    turns = json.loads(path.with_suffix(".json").read_text())
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    dur = len(audio) / sr
    print(f"{path.name}: {dur:.1f}s, {len(turns)} turns, "
          f"{len({t['speaker'] for t in turns})} speakers")

    from whisperlivekit.diarization.sortformer_backend import SortformerDiarization
    if sr != 16000:
        raise ValueError("Expected 16 kHz audio")
    shared = SortformerDiarization()
    t0 = time.time()
    segs_c = canonical(shared, audio)
    tc = time.time() - t0
    t0 = time.time()
    segs_w = asyncio.run(wlk(shared, audio))
    tw = time.time() - t0
    print(f"canonical RTF {tc/dur:.2f}   wlk RTF {tw/dur:.2f}")
    score("canonical", segs_c, turns, dur)
    score("wlk", segs_w, turns, dur)


if __name__ == "__main__":
    main()
