#!/usr/bin/env python3
"""Build multi-speaker Persian conversations with exact turn ground truth.

Uses the per-speaker clips in ~/Hobbies/voiceprint/fa_clips (16 kHz mono) and
stitches alternating turns with short gaps. Each clip is silence-trimmed first,
so a turn's [start, end] really is speech -- otherwise DER punishes the
diarizer for correctly marking lead-in/trailing silence as non-speech.

Outputs (WAV + JSON with [{start, end, speaker, source}]):
  testdata/diar-2spk.wav        two speakers, 3 turns each (~22 s)
  testdata/diar-2spk-short.wav  first 4 turns of the above (~14 s)
  testdata/diar-4spk.wav        four speakers, 2 turns each (~35 s)
  testdata/diar-4spk-long.wav   four speakers, long contiguous turns (~45 s)
  testdata/diar-4spk-2min.wav   four speakers, 4 cycles (~80 s)
"""
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

CLIPS = Path("/home/ameer/Hobbies/voiceprint/fa_clips")
OUT = Path("testdata")
SR = 16000
GAP = 0.45
TRIM_DB = 30


def load_trimmed(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: {sr} Hz"
    audio = audio.mean(axis=1)
    trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_DB)
    if len(trimmed) < sr * 0.2:  # keep at least ~200 ms
        trimmed = audio
    return trimmed


def write(name, turns_and_audio):
    parts, turns, t = [], [], 0.0
    for speaker, source, audio in turns_and_audio:
        turns.append({"start": round(t, 3), "end": round(t + len(audio) / SR, 3),
                      "speaker": speaker, "source": source})
        parts += [audio, np.zeros(int(GAP * SR), dtype=np.float32)]
        t += len(audio) / SR + GAP
    full = np.concatenate(parts)
    sf.write(OUT / f"{name}.wav", full, SR)
    (OUT / f"{name}.json").write_text(json.dumps(turns, indent=1))
    real_speech = sum(x["end"] - x["start"] for x in turns)
    print(f"{name}: {len(full)/SR:5.1f}s total, {real_speech:5.1f}s speech, "
          f"{len(turns)} turns, {len({x['speaker'] for x in turns})} speakers")


def clips_for(speaker, n=None):
    files = sorted((CLIPS / speaker).glob("*.wav"))
    picked = files if n is None else files[:n]
    return [(speaker, f.name, load_trimmed(f)) for f in picked]


def main():
    OUT.mkdir(exist_ok=True)
    a, b, c, d = "spk_25692", "spk_27383", "spk_52762", "spk_77323"

    ca, cb = clips_for(a, 3), clips_for(b, 3)
    two = [ca[0], cb[0], ca[1], cb[1], ca[2], cb[2]]
    write("diar-2spk", two)
    write("diar-2spk-short", two[:4])

    write("diar-4spk", [ca[0], cb[0], clips_for(c, 1)[0], clips_for(d, 1)[0],
                        ca[1], clips_for(c, 2)[1], clips_for(d, 2)[1], cb[1]])

    def long_turn(speaker, k):
        parts = clips_for(speaker, k)
        audio = np.concatenate([p[2] for p in parts])
        return (speaker, "+".join(p[1] for p in parts), audio)

    write("diar-4spk-long", [long_turn(a, 2), long_turn(b, 2), long_turn(c, 2),
                             long_turn(d, 2), ca[2], cb[2]])

    cycle = []
    for rnd in range(4):
        for speaker in (a, b, c, d):
            cycle.append(clips_for(speaker)[rnd % len(clips_for(speaker))])
    write("diar-4spk-2min", cycle)


if __name__ == "__main__":
    main()
