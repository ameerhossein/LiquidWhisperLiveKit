# BuzzASR/persian on WhisperLiveKit

Run [`BuzzASR/persian`](https://huggingface.co/BuzzASR/persian) — a Persian
full fine-tune of `openai/whisper-large-v3` with a **replaced, Persian-specific
tokenizer** — as a live streaming ASR backend inside
[WhisperLiveKit](https://github.com/QUENTINFUXA/WHISPERLIVEKIT).

It works with the stock `faster-whisper` backend and the LocalAgreement policy.
WhisperLiveKit itself is **unmodified**; all work lives in `scripts/`.

```bash
./run.sh              # CPU  -> http://127.0.0.1:8000  (int8 CTranslate2)
DEVICE=gpu ./run.sh   # CUDA -> same command (float16 CTranslate2)
```

---

## Verified output

Offline, 4 s clean clip:

> کلید را به در انداخت و وارد آپارتمان شد

Offline, 17.5 s (four clips concatenated), with word timings:

> رفتارهای سمی اطرافمان میتواند احساس گناه را تقویت کند … کلید را به در انداخت
> و وارد آپارتمان شد … پدربزرگ لبخند زد و گفت این همان دانیست که اجدادمان داشتند

`words`: `رفتارهای(0.00) سمی(1.28) اطرافمان(1.68) … داشتند(16.44)` — timings are
real, not collapsed.

Live WebSocket session (9.6 s, streamed at 1× real time):

```
[ 30.36s] رفتارهاى سمی اطرافى میتواند احساس گناه را تقویت کند
[ 57.92s] رفتارهاى سمی اطرافى میتواند احساس گناه را تقویت کند را به در انداخت و وارد آپارتمان شد
[ 57.92s] ready_to_stop
```

REST on the same file returns the same text with 17 word timings. The transcript
is right; the *latency* is CPU-bound (see [Performance](#performance)).

---

## Quick start

This project runs inference with prepared CTranslate2 models. Training, tokenizer
repair, export, and CT2 conversion live in the separate `stt` project.

For a local model:

```bash
MODEL_DIR=/path/to/prepared-ct2 ./setup.sh
MODEL_DIR=/path/to/prepared-ct2 ./run.sh
```

Without `MODEL_DIR`, CPU uses `models/buzzasr-persian-ct2-int8` and CUDA uses
`models/buzzasr-persian-ct2-fp16`. Copy a prepared export there before setup.

For an HF repo containing a **CT2 artifact** (`model.bin` and its configs):

```bash
# Private repos: save credentials once, or supply HF_TOKEN in the environment.
source env.sh
.venv/bin/hf auth login
HF_REPO_ID=owner/model-ct2 HF_REVISION=main ./run.sh
# An artifact nested in a run mirror:
HF_REPO_ID=owner/run-mirror HF_SUBFOLDER=runs/RUN_ID/ct2 \
  HF_REVISION=COMMIT_SHA DEVICE=gpu ./run.sh
```

Use the same HF variables with `./setup.sh` on a fresh checkout; `HF_TOKEN`
works before the CLI is installed. Downloads use the project HF cache and reuse
saved authentication. Set `HF_HUB_OFFLINE=1` to use cached snapshots only. Set
one of `MODEL_DIR` or `HF_REPO_ID`. HF repositories containing Transformers
weights (`model.safetensors`) must be converted in `stt` before use here.
`HF_SUBFOLDER` points to the CT2 directory, not the merged HF weights directory.

```bash
./scripts/verify.sh                       # offline + REST + WebSocket acceptance
./run.sh                                 # serve at http://127.0.0.1:8000
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@testdata/verify-fa.wav -F response_format=verbose_json
./run.sh --pcm-input
.venv/bin/python scripts/stream_test.py testdata/verify-fa.wav \
  --url ws://127.0.0.1:8000/asr --realtime 1.0
.venv/bin/python scripts/transcribe.py testdata/verify-fa.wav --words
HF_REPO_ID=owner/model-ct2 .venv/bin/python scripts/transcribe.py audio.wav
```

Local models may also be selected with the offline tool's `--model` argument.
For a different fine-tune, pass its reference audio/text to acceptance checks;
the default fixture's exact transcript was measured with the original BuzzASR model.

## Decoding behavior

### Decoding defaults fight this model

WhisperLiveKit hardcodes `beam_size=5` and leaves faster-whisper's temperature
fallback at `[0.0 … 1.0]`. On *partial* buffers (every streaming update) the
decoded text looks repetitive, the compression-ratio check fails, and the
segment is re-decoded up to six more times:

| 5 s buffer, greedy | time | text |
|---|---|---|
| temperature fallback on (default) | 102 s | looping tail |
| single temperature | 30 s | coherent |

Beam 5 additionally latches onto repetition loops. The model card recommends
`num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2`; on a mid-sentence
slice that is 22 s coherent vs 38 s looping.

### Prompt carry-over poisons the stream

WhisperLiveKit feeds the committed transcript back as `initial_prompt` and keeps
`condition_on_previous_text=True`. For this model that is destructive: the
buffer already contains the audio it is being told about, so a single repeated
word snowballs. Concretely, the second update of a two-sentence clip came out as
`… ب bbbbnbbbcbbbbnbn` with prompts on, and as the correct sentence with prompts
off. `scripts/serve.py` therefore defaults both to **off** (`WLK_USE_PROMPT=1`
restores upstream behaviour).

`scripts/serve.py` is a drop-in `wlk` launcher that applies all of the above by
subclassing `FasterWhisperASR`; WhisperLiveKit's own code is not edited.
Knobs: `WLK_BEAM`, `WLK_TEMPERATURE`, `WLK_NO_REPEAT_NGRAM`,
`WLK_REPETITION_PENALTY`, `WLK_NO_SPEECH_THRESHOLD`,
`WLK_CONDITION_ON_PREVIOUS`, `WLK_USE_PROMPT`, `WLK_VAD_FILTER`
(faster-whisper's internal silence stripping; helps accuracy on buffers that
WhisperLiveKit pads with silence, at the cost of ~1 extra call per buffer).

---

## Performance

Measured on this machine: AMD Ryzen 5 7430U (6 cores / 12 threads, 15 W), 15 GB
RAM, no GPU. CTranslate2 `int8_float32`, `cpu_threads` default.

| Case | Time | Notes |
|---|---|---|
| 17.5 s full clip, offline | 11.3 s | RTF 0.64, greedy + words |
| 17.5 s full clip, beam 5 | 17.7 s | RTF 1.0 |
| 4.1 s clean clip | ~10 s | floor is the 30 s padded encoder |
| 5 s mid-sentence slice, greedy | ~22 s | worst case, loop-prone |
| 9.6 s live stream, `ASR_COALESCE=6` | ~58 s to final | both updates correct |
| 17.5 s live stream, `WLK_VAD_FILTER=1` | ~190 s processing (RTF 10.9), 9 calls | correct to sentence 3, then filler |

The important number is the **~10 s floor per ASR call**: Whisper always encodes
a padded 30 s window, and large-v3's encoder needs ~8–9 s on this CPU. Streaming
calls ASR once per pause (WhisperLiveKit re-transcribes a rolling buffer), so
live transcription here runs at roughly 0.2–0.5× real time and settles at a
10–20 s lag. It is usable as a demo, **not** as real-time dictation.

What helps on CPU:

* `ASR_COALESCE=6` (default in `run.sh`) — fewer, larger encoder passes;
* keep buffers short (`--buffer_trimming_sec 10`) so degenerate buffers are dropped sooner;
* nothing else moves the needle much — beam 1 vs 5 is only ~10 %.

**Use a GPU for interactive use.** See [`docs/CUDA_PLAN.md`](docs/CUDA_PLAN.md)
for the host setup, the decoding matrix to re-test quality, and the pitfalls.
GPU speed and latency depend on the hardware and have not been measured here; run the acceptance check and latency/quality tools on the target host. Precision is
not the reason: on the 17.5 s clip, int8/fp32 differ only by "دانیست" vs
"دانی است", and fp16-weights-in-fp32 on CPU costs 22.7 s vs 13.8 s for int8.
The GPU's quality advantage is indirect — you can afford beam 5, temperature
fallback and 0 s coalescing instead of the CPU-safe settings.

---

## Files

| Path | What it is |
|---|---|
| `env.sh` | venv + `HF_HOME`/`UV_CACHE_DIR` inside the project |
| `run.sh` | server launcher: `DEVICE`, `HOST`, `PORT`, `ASR_COALESCE`, `REST_TIMEOUT`, `LOG_LEVEL` |
| `scripts/serve.py` | `wlk` launcher with the decoding settings above |
| `scripts/transcribe.py` | offline transcription + real-time factor |
| `scripts/stream_test.py` | streams a WAV into `/asr` over WebSocket (PCM) |
| `scripts/verify.sh` | offline + REST + streaming end-to-end check |
| `scripts/bench_cpu.py`, `bench_encoder.py` | thread/beam benchmarks |
| `models/buzzasr-persian-ct2-int8/` | CTranslate2 model for CPU (1.5 GB) |
| `models/buzzasr-persian-ct2-fp16/` | CTranslate2 model for CUDA (2.9 GB) |
| `wlk/` | WhisperLiveKit clone, unmodified |
| `wheels/`, `.cache/`, `.venv/` | cached runtime dependencies and HF downloads |

## Environment notes

* The venv is Python **3.12** (WhisperLiveKit requires `<3.14`; the system has 3.14).
* Dependencies were installed **fully offline** from a copied uv wheel cache after
  PyPI fetches kept timing out; `scripts/build_wheels_from_cache.py` can rebuild
  wheels from uv's unpacked cache if needed.
* `wlk serve` prints `Model: base` in its banner regardless of `--model_dir`; trust
  the `Loading Whisper … using backend faster-whisper` log line and `/health`.
* Warmup uses a local Persian clip (`testdata/warmup-fa.wav`) instead of
  WhisperLiveKit's 60 kB GitHub download, so startup works offline.

## Diarization (2 speakers good, 4 speakers mediocre)

WhisperLiveKit's only CPU-viable diarizer is NeMo **Streaming Sortformer v2**
(`nvidia/diar_streaming_sortformer_4spk-v2`, 471 MB); Diart pins `torch<2.9`.

```bash
source env.sh
uv pip install "nemo_toolkit[asr]>=3.0,<4"        # 86 packages, already done here
.venv/bin/python scripts/make_diar_test.py         # ground-truth clips from ~/fa_clips
.venv/bin/python scripts/eval_diar.py testdata/diar-2spk.wav
./run.sh --diarization --diarization-backend sortformer \
         --sortformer-max-speakers 2 --pcm-input
```

`eval_diar.py` drives the same `SortformerDiarizationOnline` object the server
uses, feeds the WAV in 0.5 s chunks, and scores the emitted `SpeakerSegment`s
against the known turn boundaries (frame-level DER, 10 ms frames, optimal
hypothesis→reference speaker mapping).

| Mode | Clip | Speakers | Predicted | DER (miss / false-alarm / confusion) |
|---|---|---|---|---|
| streaming | `diar-2spk` 22.3 s | 2 | 2 | **12.1 %** (0.0 / 12.0 / 0.2) |
| streaming | `diar-4spk-2min` 67.3 s | 4 | 4 | **48.0 %** (0.0 / 11.5 / 36.5) — two identities swapped |
| offline, onset 0.5 | `diar-4spk-2min` | 4 | 4 | **45.2 %** (12.7 / 10.0 / 22.5) — 1:1 identity mapping |
| offline, onset 0.1 | `diar-4spk-2min` | 4 | 4 | **43.8 %** (4.0 / 11.3 / 28.4) — 1:1 identity mapping |

Methodology note: an earlier version of this table reported 36–74 % DER because
`make_diar_test.py` labelled each clip's *entire file* as speech, including its
lead-in/trailing silence — the diarizer was being punished for correctly treating
silence as non-speech. The script now silence-trims every clip (30 dB), so a
turn's `[start, end]` is speech. Numbers above use that corrected reference.

Reading the numbers:

* **2 speakers is good**: 12 % DER with perfect attribution; the 12 % false alarm
  is the diarizer labelling the 0.45 s gaps between turns as a speaker, because
  WhisperLiveKit's `_process_predictions` takes an `argmax` over speaker channels
  with **no silence threshold** — there is no "no speaker" class in its output.
* **4 speakers is mediocre**: all four channels appear, but streaming swaps two
  identities (confusion 36.5 %). The **offline** NeMo runner (`model.diarize()`,
  full-context, post-processed) keeps a correct 1:1 mapping and halves confusion,
  at the cost of dropping ~5–13 % of speech.
* A one-to-one mapping row of `{1:1, 2:3, 3:2, 4:4}` in the logs is the swap; it
  is an identification error, not a missed speaker.

Live check on `diar-2spk-short` (10.1 s, 4 turns): emitted lines carried the right
speaker for their time range — `S1 [0.00-1.98]`, `S2 [2.56-3.14]` — matching turns
1 and 2 exactly, while offline the same clip scored 4/4 turns at 100 % purity.
The remaining live defect was the **ASR**, not the diarizer: under CPU load it
committed only 2 of 4 turns before the client timeout and its text included
hallucinated filler, so one long merged line looked like "everything is S1" even
though its timestamps sat inside speaker 1's first turn.

A GPU does **not** change DER (float32 on both sides); what it buys is the offline
or rolling-window re-diarization pass, which fixes identity swaps for 4 speakers.

## Known limitations

* **Silence hallucination.** With prompt carry-over enabled, buffers containing
  silence made the model invent fluent Persian (or count numbers). Disabling
  prompts plus `no_speech_threshold=0.8` fixed it in testing, but any
  Whisper-family model can still hallucinate on long silences — verify output on
  your own audio before trusting it unattended.
* **Late-buffer filler.** On longer streams the last update can append
  YouTube-style filler ("در این مقاله قصد داریم …") after the real speech, where
  the buffer is mostly trailing silence. `WLK_VAD_FILTER=1` removes most of it.
* **Streaming is ~10× slower than real time on this CPU.** A 17.5 s stream made
  6–9 ASR calls totalling ~190 s (RTF 10.9, p95 latency 48 s). Treat live use as
  a correctness demo; use a GPU for anything interactive.
* **REST is slow on CPU** — a 17.5 s upload takes minutes; `run.sh` raises
  `--rest-timeout` to 900 s. The WebSocket path is the one to demo.
* **Persian only.** The tokenizer has no other language tokens; `--language fa`
  is required. English audio produces fluent Persian hallucination (by design —
  it has no way to say "this isn't Persian").

## Reproducible runtime setup

`./setup.sh` pins WhisperLiveKit to commit
`363e4f6d029694d9c81ae548beddd9d3c88a3637` and installs the Python 3.12 runtime
from `requirements.lock`. It validates a prepared local or HF CT2 artifact;
it does not download raw checkpoints or build models. Torch and torchaudio
use a matched 2.11.0 pair, installed separately for CPU or CUDA. For CUDA, set
`DEVICE=gpu` and, when needed, `TORCH_INDEX_URL`. `uv`, `git`, `ffmpeg`, and
`curl` are required. Optional diarization dependencies are installed separately.

`env.sh` derives its location from the checkout, so the project can be moved.
`DEVICE=cpu` explicitly selects CPU/int8_float32; `DEVICE=gpu` or `cuda` selects
CUDA/float16 and fails if CUDA is unavailable. Metal is unsupported by this
CTranslate2 adaptation. Launchers check required model files before loading.

All transcription and benchmark scripts share `WLK_PROFILE=tuned` (default) or
`WLK_PROFILE=baseline`. Existing WLK decoding overrides still apply. Use
`--profile` on the offline and benchmark scripts. Resolved settings are printed.
`WLK_USE_PROMPT=0` disables rolling transcript history while preserving explicit
REST `prompt` / WebSocket `context`. `WLK_USE_PROMPT=1` restores both, and the
baseline server profile enables history by default. `WLK_CPU_THREADS=0` uses
CTranslate2's default; set a positive value to tune CPU work. `WLK_NUM_WORKERS`
configures the model worker pool; upstream session locking still serializes
calls to the shared ASR model. Server INFO logs include ASR call count and time.
The tuned launcher defaults to `WLK_VAC=0`, preserving continuous audio instead
of forcing inference on subsecond pause fragments. This fixed a reproduced
hallucinated paragraph on a 0.67-second fragment. Use `WLK_VAC=1` to explicitly
compare upstream pause splitting; the baseline profile retains that behavior.

Run `./scripts/verify.sh` for offline, REST and WebSocket acceptance against the
checked-in `testdata/verify-fa.wav` and its expected Persian transcript. A custom
file must produce text and valid word timings; set `EXPECTED_FILE` to also
check its reference. HTTP errors, missing finalization and timeouts fail the
check. `REST_TIMEOUT` sets the common REST/WS finish budget; results and server
logs are retained under `.cache/verify-*/`.

`stream_test.py --json-output result.json` records time to first partial,
first committed text, and finalization. `transcribe.py --json-output result.json`
records text, word boundaries, ASR time, RTF and peak process RSS. Compare either with a reference JSON:

```bash
.venv/bin/python scripts/eval_quality.py reference.json result.json
```

References contain `{"text": "..."}` and optionally manually annotated `words`
with `word`, `start`, and `end` fields. The tool reports raw and normalized
Persian CER/WER. Matching word sequences also get boundary MAE/p95; inherited
large-v3 attention heads still require manual Persian alignment validation.
Normalization changes evaluation only, preserving display spelling.

Diarization comparison now retains cumulative prediction history, feeds tensors
to the model device, rejects non-16-kHz input, drains all full model chunks, and
pads/clips the final tail in both drivers. Historical comparison scores must be
rerun. DER uses 10 ms frames, no boundary collar, and one speaker per frame;
overlapping speech is unsupported. Both drivers retain the model integration's
argmax speaker policy, including during silence.

Regression checks (install `pytest==9.1.1 pytest-asyncio==1.4.0` in a test environment):

```bash
source env.sh
.venv/bin/python -m pytest tests wlk/tests/test_asr_coalescing.py wlk/tests/test_session_context.py -q
```

To regenerate the runtime lock on Python 3.12, compile `requirements.txt` and
the pinned `wlk/pyproject.toml` with torch/torchaudio constrained to 2.11.0, then
exclude their two entries from the runtime lock (setup installs that pair per
device). Optional diarization dependencies are separate from this ASR lock.
