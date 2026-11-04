# CUDA migration plan

Status: **not executed here** — this machine has no NVIDIA GPU (`torch.cuda.is_available()`
is False). Everything below is the same pipeline, with the float16 CTranslate2
build, intended to be run on a CUDA box and compared against the CPU numbers in
`README.md`.

The plan has two goals:

1. make the live pipeline usable in real time (CPU is ~0.1–0.5× real time);
2. settle the open quality question — how much of the CPU-era degradation came
   from *decoding settings we chose to keep up*, rather than from the model.

---

## 0. What actually changes

| | CPU (current, measured) | GPU (target) |
|---|---|---|
| ASR model | `models/buzzasr-persian-ct2-int8` | `models/buzzasr-persian-ct2-fp16` |
| CTranslate2 compute | `int8_float32` | `float16` (auto-detected) |
| Torch | `2.14.1+cpu` | CUDA build matching the driver |
| `ASR_COALESCE` | `6` | `0` |
| `WLK_BEAM` | `1` | `5` |
| `WLK_TEMPERATURE` | `0.0` (fallback off) | empty → fallback list `[0.0 … 1.0]` |
| Expected ASR speed | ~10 s per call (30 s padded encoder) | ~0.3–0.8 s per call |
| Expected diarizer speed | RTF ~0.7–0.8 | RTF ~0.02–0.05 |
| Expected DER | 12 % (2 spk), 48 % streaming / 45 % offline (4 spk) | **unchanged** (float32 either side) |

Precision is *not* the reason to move: on the 17.5 s clip, int8 and fp16-in-fp32
differ only by `دانیست` / `دانی است` (measured, see README). The GPU buys latency,
which in turn buys back the decoding settings above.

---

## 1. Prerequisites on the GPU host

* NVIDIA driver with CUDA 12.x support (`nvidia-smi`), ideally ≥ 535.
* CTranslate2 ≥ 4.x needs **cuDNN 9** at runtime. If the system does not have it:

  ```bash
  uv pip install --python .venv/bin/python "nvidia-cudnn-cu12==9.*"
  ```

* Python 3.12 (WhisperLiveKit requires `<3.14`), `uv`, `ffmpeg`, ~12 GB free disk
  (2.9 GB fp16 model + 3.1 GB raw weights if rebuilding + venv).
* Diarization only: ~500 MB more for the Sortformer `.nemo`.

Quick preflight, to be run **after** the venv step in §3:

```bash
.venv/bin/python -c "import torch, ctranslate2; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); \
print('ct2 cuda types', ctranslate2.get_supported_compute_types('cuda'))"
```

Must print `cuda True` and a set containing `float16`. If `ct2` raises
`Library cudnn_ops.so.9 not found`, install cuDNN 9 as above.

---

## 2. Get the project onto the GPU box

Option A — copy this working tree (recommended, keeps every fix):

```bash
rsync -a --info=progress2 \
  --exclude '.cache' --exclude 'wheels' --exclude '.venv' \
  /home/ameer/Hobbies/livebuzzasr/ gpu:~/livebuzzasr/
```

That is ~7.5 GB (mostly `models/`). If the link is slow, copy only what runs:

```bash
rsync -a --info=progress2 \
  --exclude '.cache' --exclude 'wheels' --exclude '.venv' \
  --exclude 'models/buzzasr-persian' --exclude 'models/buzzasr-persian-ct2-int8' \
  --exclude 'models/buzzasr-persian-hf' --exclude 'models/buzzasr-persian-tok' \
  /home/ameer/Hobbies/livebuzzasr/ gpu:~/livebuzzasr/
```

Required on the GPU box in that case: `models/buzzasr-persian-ct2-fp16/` (2.9 GB)
plus `scripts/`, `wlk/`, `run.sh`, `env.sh`, `testdata/`.

Option B — rebuild from scratch: run `./setup.sh`, then
`.venv/bin/python scripts/build_ct2.py --quantization float16 --alignment-heads-fallback large-v3`. Needs the 3.1 GB
HF download; the conversion itself takes ~40 s.

---

## 3. Environment

```bash
cd ~/livebuzzasr
source env.sh                                  # project-local HF/uv caches
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv/bin/python -r requirements.lock -e ./wlk
uv pip install --python .venv/bin/python "nemo_toolkit[asr]>=3.0,<4"   # diarization only
```

Notes

* Do **not** reuse this repo's `.venv` — it holds a CPU torch build.
* `pyproject.toml` has `cu129`/`cpu` extras but they conflict with each other;
  installing torch manually first, as above, avoids uv resolving the CPU wheel.
* `HF_HOME` is project-local (`env.sh`), so the `.nemo` will be re-downloaded on
  a fresh machine; copy `.cache/hf/hub/models--nvidia--diar_streaming_sortformer_4spk-v2`
  to skip that.

---

## 4. Verify the converted model before running anything

The reference-model fp16 directory uses the language/vocabulary patches and
the explicit large-v3 alignment fallback shown above. Source-provided
alignment heads are preserved instead:

```bash
.venv/bin/python - <<'PY'
import json, tokenizers
d = "models/buzzasr-persian-ct2-fp16"
cfg = json.load(open(f"{d}/config.json"))
print("lang_ids", cfg["lang_ids"], "heads", len(cfg["alignment_heads"]))
vocab = json.load(open(f"{d}/vocabulary.json"))
print("empty token at", vocab.index(""))
tok = tokenizers.Tokenizer.from_file(f"{d}/tokenizer.json")
print("fa", tok.token_to_id("<|fa|>"), "notimestamps", tok.token_to_id("<|notimestamps|>"))
PY
```

Expected: `lang_ids [51908]`, `heads 10`, empty token at `220`, `fa 51908`,
`notimestamps 51972`. The head count can legitimately differ when source heads
are present. To rebuild the reference model with its explicit fallback, run
`scripts/build_ct2.py --quantization float16 --alignment-heads-fallback large-v3`.

---

## 5. Run

Start conservative (GPU-safe, lowest latency):

```bash
DEVICE=gpu ASR_COALESCE=0 REST_TIMEOUT=300 WLK_BEAM=5 WLK_TEMPERATURE= ./run.sh
```

`WLK_TEMPERATURE=` (empty) restores faster-whisper's temperature fallback; the
launcher's docstring lists every knob. Open <http://127.0.0.1:8000>.

```bash
# streaming, PCM mode
DEVICE=gpu ASR_COALESCE=0 WLK_BEAM=5 WLK_TEMPERATURE= ./run.sh --pcm-input
.venv/bin/python scripts/stream_test.py testdata/persian-10s.wav \
    --url ws://127.0.0.1:8000/asr --realtime 1.0
```

---

## 6. The quality experiment (the real point of the move)

Run the **same audio** through these four settings and keep the transcripts.
`testdata/persian-15s.wav` is the reference clip used for all CPU numbers.

| # | `WLK_BEAM` | `WLK_TEMPERATURE` | `ASR_COALESCE` | `WLK_USE_PROMPT` | Question it answers |
|---|---|---|---|---|---|
| A | 1 | 0.0 | 6 | 0 | CPU baseline (already measured) |
| B | 5 | 0.0 | 0 | 0 | is beam 5 worth it? |
| C | 5 | *(empty)* | 0 | 0 | does fallback recover loops/filler? |
| D | 5 | *(empty)* | 0 | 1 | does prompt carry-over help once it is fast? |

For each, record:

* transcript of `testdata/persian-15s.wav` (and a few of your own clips),
* CER against a small hand-checked reference (5–10 utterances is enough):

  ```bash
  uv pip install --python .venv/bin/python jiwer
  .venv/bin/python - <<'PY'
  import jiwer
  ref = "..."; hyp = "..."
  print("CER", jiwer.cer(ref, hyp), "WER", jiwer.wer(ref, hyp))
  PY
  ```

### Interpretation gate

* If **B/C are no better than A** on CER → the CPU settings were not costing
  quality; the residual errors are model behaviour. Stop tuning decoding.
* If **C beats B** → keep fallback on; the loop-prone partial buffers were the
  problem and the CPU was silently living with them.
* If **D beats C** → prompt carry-over is beneficial once buffers are timely,
  i.e. it was a latency artefact, not a modelling one.

---

## 7. Diarization on GPU

```bash
DEVICE=gpu ASR_COALESCE=0 ./run.sh --diarization \
    --diarization-backend sortformer --sortformer-max-speakers 4 --pcm-input
```

Expectations, and what to verify:

* **DER should not improve from the device alone.** The diarizer runs float32 on
  CPU and GPU, so the measured 12 % (2 spk) / 48 % streaming (4 spk) should be
  reproduced. Re-score to confirm:

  ```bash
  .venv/bin/python scripts/eval_diar.py testdata/diar-2spk.wav
  .venv/bin/python scripts/eval_diar.py testdata/diar-4spk-2min.wav
  .venv/bin/python scripts/compare_diar_drivers.py testdata/diar-2spk.wav
  ```

* What should improve is *latency*: labels should arrive with the partial
  transcript instead of minutes later, and the ASR should not be starved by the
  diarizer fighting for the same cores.
* **Where GPU genuinely buys diarization quality:** the offline full-context
  runner. On CPU it costs ~45 s per 67 s file; on GPU it is ~1–2 s, so you can
  re-diarize rolling windows (say every 30–60 s) for a correct 1:1 speaker
  mapping (offline scored 45 % DER with a perfect mapping vs 48 % streaming with
  two identities swapped). Wire it as a custom backend, or call
  `SortformerEncLabelModel.diarize()` on completed audio.

  ```python
  from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
  from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams
  from dataclasses import asdict
  cfg = DiarizeConfig(postprocessing_params=asdict(
      PostProcessingParams(onset=0.1, offset=0.1)))   # 43.8 % DER, 4/4 speakers
  shared.diar_model.diarize("file.wav", batch_size=1, override_config=cfg)
  ```

* If DER does move on the same mode, that falsifies the README claim — record it
  and update the diarization section.

---

## 8. Acceptance checklist

```bash
# 1. offline, expect RTF << 0.1 and the Persian sentences from README
.venv/bin/python scripts/transcribe.py testdata/persian-15s.wav --words

# 2. full stack: offline + REST + streaming
./scripts/verify.sh

# 3. REST latency on 17.5 s, expect a few seconds, not minutes
time curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@testdata/persian-15s.wav -F response_format=verbose_json

# 4. live, expect first update < ~2 s after speech starts
.venv/bin/python scripts/stream_test.py testdata/diar-2spk-short.wav \
  --url ws://127.0.0.1:8000/asr --realtime 1.0

# 5. diarization unchanged
.venv/bin/python scripts/eval_diar.py testdata/diar-4spk-2min.wav
```

Fill in and commit:

| Metric | CPU (measured) | GPU (fill in) |
|---|---|---|
| ASR RTF, 17.5 s clip | 0.64 (greedy), 1.0 (beam 5) | |
| Per-ASR-call latency | ~10 s floor | |
| Live first update, 10 s clip | ~28 s | |
| REST 17.5 s | ~5 min | |
| Diarization RTF | 0.69–0.80 | |
| DER 2 spk / 4 spk | 36 % / 56–74 % | |
| CER, 17.5 s clip | *(fill from experiment A–D)* | |

---

## 9. Pitfalls

* **`torch.cuda.is_available()` False** → the CPU wheel got installed. Reinstall
  from the cu12x index; check `nvidia-smi` works.
* **CTranslate2 falls back with** *"float16 … not supported … converted to
  float32"* → it is running on CPU. Check the model dir (`-fp16`, not `-int8`)
  and that CUDA is visible. Do **not** point GPU runs at the int8 directory.
* **cuDNN missing** → `Library cudnn_ops.so.9 not found`; install
  `nvidia-cudnn-cu12==9.*`.
* `wlk` prints `Model: base` in the banner regardless of `--model_dir`; trust
  `/health` and the `Loading Whisper … backend faster-whisper` log line.
* NeMo emits harmless `lhotse`/`OneLogger` warnings and cannot write `~/.lhotse`
  under a restrictive sandbox; diarization still works.
* First `--diarization` start loads a 471 MB model; expect 10–30 s extra startup.
* Transcripts are **not** bit-identical across precisions: expect occasional
  spacing variants (`دانیست` vs `دانی است`), not different content.
* Keep `env.sh` so caches stay inside the project (sandboxed hosts otherwise
  fail writing to `~/.cache`).

## 10. Rollback

Nothing here mutates the CPU setup: `run.sh` selects the model directory from
`DEVICE`, and the CPU path (`DEVICE=cpu`, int8, coalescing 6) stays as measured.
To go back, just drop `DEVICE=gpu`.

The adaptation now enforces DEVICE explicitly and rejects CPU fallback for a requested CUDA device. See the README reliability section for pinned setup and strict acceptance checks. Historical canonical diarization comparisons need to be rerun after the cumulative-history fix.
