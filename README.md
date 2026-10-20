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

```bash
source env.sh                                    # venv + project-local caches
.venv/bin/python scripts/prepare_tokenizer.py    # rebuild corrected tokenizer
.venv/bin/python scripts/build_ct2.py            # HF -> int8 (CPU) + float16 (GPU)
./scripts/verify.sh                              # offline + REST + WebSocket
./run.sh                                         # serve
```

Then open <http://127.0.0.1:8000>, or:

```bash
# REST (OpenAI-compatible). Slow on CPU: --rest-timeout is raised by run.sh.
curl http://127.0.0.1:8000/v1/audio/transcriptions \
     -F file=@testdata/persian-10s.wav -F response_format=verbose_json

# WebSocket streaming: sends raw PCM, so start the server with --pcm-input
./run.sh --pcm-input
.venv/bin/python scripts/stream_test.py testdata/persian-10s.wav \
     --url ws://127.0.0.1:8000/asr --realtime 1.0

# Offline file + real-time factor
.venv/bin/python scripts/transcribe.py testdata/persian-10s.wav --words
```

---

## Why this needs fixing at all

The checkpoint looks like a drop-in Whisper model, but three details silently
break every generic Whisper integration. Each has a concrete symptom.

### 1. The tokenizer shadows its own special tokens

`vocab.json` has 51,865 BPE entries whose **first 1,609 are the special and
timestamp strings** (`<|endoftext|>`=0, `<|startoftranscript|>`=1,
`<|notimestamps|>`=7, `<|0.00|>`=108 … `<|30.00|>`=1608). The same strings also
exist in `added_tokens.json` at **51,865…53,473**, which is what the checkpoint
actually uses:

```
bos/eos/pad=51865  decoder_start=51866  <|fa|>=51908
<|transcribe|>=51968  <|notimestamps|>=51972  <|0.00|>=51973
```

Hugging Face resolves a string against the BPE vocab first, so
`convert_tokens_to_ids("<|fa|>")` returns **51906**, `<|endoftext|>` returns
**0**, and `timestamp_begin` becomes **8**. In practice: empty transcripts
(faster-whisper drops every token `< eot`), no timestamps, wrong prompt.

`scripts/prepare_tokenizer.py` keeps all 51,865 BPE ids (so text ids
1,609…51,864 — and therefore the embedding rows — stay put) but renames the
1,509 shadowed placeholders, then re-adds the real specials in ascending id
order. The result is exactly 53,474 tokens with every id matching
`generation_config.json`; the script asserts that and round-trips a Persian
sentence.

### 2. CTranslate2 thinks the model is English-only

CTranslate2 decides "multilingual" with a heuristic:

```cpp
_is_multilingual = vocabulary.to_id("") != vocabulary.unk_id();
```

Whisper's original multilingual vocabularies contain an empty-string token;
BuzzASR's Persian BPE does not. faster-whisper then logs *"The current model is
English-only but the language parameter is set to 'fa'"* and drops the
`<|fa|><|transcribe|>` prompt.

`scripts/build_ct2.py` writes `""` into `vocabulary.json[220]` after conversion.
Id 220 is a dead placeholder row in this checkpoint, and faster-whisper decodes
with `tokenizer.json`, not this file, so nothing else is affected.

### 3. `lang_to_id` and alignment heads never reach the converter

* The converter reads `generation_config.lang_to_id`, but transformers 5 drops
  unknown `GenerationConfig` fields on load, so `lang_ids` ends up empty and
  `detect_language()` would fail. The build script writes `lang_ids: [51908]`.
* BuzzASR ships **no alignment heads**. The converter's fallback ("last half of
  the layers × every head" = 320 heads) produced word timings that collapsed
  onto identical values. The build script installs whisper-large-v3's 10
  published heads, which fixes timing and removes the DTW overhead.

### 4. Decoding defaults fight this model

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

### 5. Prompt carry-over poisons the stream

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
`WLK_CONDITION_ON_PREVIOUS`, `WLK_USE_PROMPT`.

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

The important number is the **~10 s floor per ASR call**: Whisper always encodes
a padded 30 s window, and large-v3's encoder needs ~8–9 s on this CPU. Streaming
calls ASR once per pause (WhisperLiveKit re-transcribes a rolling buffer), so
live transcription here runs at roughly 0.2–0.5× real time and settles at a
10–20 s lag. It is usable as a demo, **not** as real-time dictation.

What helps on CPU:

* `ASR_COALESCE=6` (default in `run.sh`) — fewer, larger encoder passes;
* keep buffers short (`--buffer_trimming_sec 10`) so degenerate buffers are dropped sooner;
* nothing else moves the needle much — beam 1 vs 5 is only ~10 %.

**Use a GPU for interactive use.** With `DEVICE=gpu` (float16, CUDA), the same
pipeline is ~20–50× faster and latency drops to well under a second.

---

## Files

| Path | What it is |
|---|---|
| `env.sh` | venv + `HF_HOME`/`UV_CACHE_DIR` inside the project |
| `run.sh` | server launcher: `DEVICE`, `HOST`, `PORT`, `ASR_COALESCE`, `REST_TIMEOUT`, `LOG_LEVEL` |
| `scripts/serve.py` | `wlk` launcher with the decoding settings above |
| `scripts/prepare_tokenizer.py` | fixes shadowed special tokens, validates ids |
| `scripts/build_ct2.py` | HF → CTranslate2 int8/float16 + the three config patches |
| `scripts/fetch.py` | resumable multi-connection downloader (HF signed URLs expire) |
| `scripts/transcribe.py` | offline transcription + real-time factor |
| `scripts/stream_test.py` | streams a WAV into `/asr` over WebSocket (PCM) |
| `scripts/verify.sh` | offline + REST + streaming end-to-end check |
| `scripts/hf_reference.py` | plain-transformers reference transcription |
| `scripts/bench_cpu.py`, `bench_encoder.py` | thread/beam benchmarks |
| `models/buzzasr-persian/` | raw HF files (3.1 GB safetensors) |
| `models/buzzasr-persian-tok/` | corrected tokenizer |
| `models/buzzasr-persian-ct2-int8/` | CTranslate2 model for CPU (1.5 GB) |
| `models/buzzasr-persian-ct2-fp16/` | CTranslate2 model for CUDA (2.9 GB) |
| `wlk/` | WhisperLiveKit clone, unmodified |
| `wheels/`, `.cache/`, `.venv/` | offline build artifacts |

## Environment notes

* The venv is Python **3.12** (WhisperLiveKit requires `<3.14`; the system has 3.14).
* Dependencies were installed **fully offline** from a copied uv wheel cache after
  PyPI fetches kept timing out; `scripts/build_wheels_from_cache.py` can rebuild
  wheels from uv's unpacked cache if needed.
* `wlk serve` prints `Model: base` in its banner regardless of `--model_dir`; trust
  the `Loading Whisper … using backend faster-whisper` log line and `/health`.
* Warmup uses a local Persian clip (`testdata/warmup-fa.wav`) instead of
  WhisperLiveKit's 60 kB GitHub download, so startup works offline.

## Known limitations

* **Silence hallucination.** With prompt carry-over enabled, buffers containing
  silence made the model invent fluent Persian (or count numbers). Disabling
  prompts plus `no_speech_threshold=0.8` fixed it in testing, but any
  Whisper-family model can still hallucinate on long silences — verify output on
  your own audio before trusting it unattended.
* **REST is slow on CPU** — a 17.5 s upload takes minutes; `run.sh` raises
  `--rest-timeout` to 900 s. The WebSocket path is the one to demo.
* **Persian only.** The tokenizer has no other language tokens; `--language fa`
  is required. English audio produces fluent Persian hallucination (by design —
  it has no way to say "this isn't Persian").

## Using another BuzzASR language

All BuzzASR checkpoints share this layout. Change `--src`/`--tok` in
`scripts/prepare_tokenizer.py`, then `--src`/`--tok`/`--out-prefix` in
`scripts/build_ct2.py`, and update three constants: the language id
(`<|xx|>`), the `lang_ids` patch, and `--language` in `run.sh`.
