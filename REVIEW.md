# BuzzASR / WhisperLiveKit review

## Fix status

All ten code findings below have been addressed. The original findings are
retained as the audit record; their line numbers refer to the reviewed snapshot.

| Finding | Resolution |
| --- | --- |
| 1–2: corrupt downloads | Append-only parts, exact range checks, resource/layout manifest, bounded retries, writer locking, checksum verification, atomic file publication |
| 3: incomplete setup | Pinned upstream/runtime/model assets, integrity-checked bootstrap, tool checks, checked-in speech fixture |
| 4–5: runtime selection/paths | Explicit CPU/CUDA settings, unsupported-device errors, relocatable environment, model readiness checks |
| 6: diarization comparison | Cumulative history, device-aware tensors, complete chunk draining, tail padding/clipping, sample-rate validation |
| 7: discarded prompts | Explicit session terminology survives while rolling transcript history remains independently controlled |
| 8: false verification success | HTTP/protocol/timeout failures propagate; exact reference text, word timings and finalization are required |
| 9: destructive conversion | Build/validate in staging; preserve the old model and restore it if publication fails |
| 10: inconsistent decoding | Shared tuned/baseline profiles in server, offline transcription and benchmarks |

Additional changes include CPU thread/worker controls, lighter model validation,
Persian text direction, latency/memory telemetry, CER/WER and manual boundary
evaluation, and adaptation regression tests. The tuned server defaults to
continuous audio (`WLK_VAC=0`) after real acceptance testing reproduced a
hallucinated paragraph from upstream pause splitting of a 0.67-second fragment.
`WLK_VAC=1` enables comparison with that upstream behavior.

The existing raw checkpoint had eight zero-filled gaps containing 12,321 missing
bytes. These were recovered with range requests; the repaired file now matches
the publisher's SHA256 exactly:
`fd91267e6c21939fa28d7bc7f1b887b009dd2e3489a11c3ca6244942bb7e0752`.
Setup and conversion now reject mismatched source weights.

Final verification: **53 tests passed** (adaptation regressions plus selected
upstream coalescing/session-context tests). Both int8 and fp16 artifacts were
rebuilt successfully from the repaired source and passed metadata validation.
The rebuilt CPU artifact passed real offline, REST and WebSocket acceptance
with exact Persian reference text and valid word timings. The final WebSocket
run finished in 15.99 seconds for 4.08 seconds of audio; this remains CPU-bound.
The offline source-manifest verifier checks all raw assets. Shell syntax,
Python compilation, whitespace checks, and offline dependency resolution also
passed. The nested WhisperLiveKit source checkout remains unmodified.

Acceptance outputs are in `.cache/verify-qWpE6C/`. This targeted validation does
not imply a complete upstream-suite pass; the broader runs described in the
original review were interrupted after stalls/time limits.

Manual Persian word-boundary validation, a representative noisy/mixed-language
accuracy corpus, and GPU runtime/performance measurement remain empirical work;
they cannot be established from code changes or this machine's CPU tests.
Both diarization drivers retain the integration's argmax speaker policy during
silence; overlap is unsupported and the scoring policy is documented explicitly.

## Original review

Reviewed 2026-11-04. Covers the current working tree, including uncommitted
adaptation scripts and the nested WhisperLiveKit checkout (`363e4f6`). This
review changed no application code. P1 means high priority; P2 means normal priority.

## Findings

### 1. [P1] Interrupted downloads silently skip unwritten bytes

`scripts/fetch.py:44–48, 53–54`

`SEEK_HOLE` reports filesystem allocation boundaries, not the exact number of
bytes written. Reproduction on this filesystem: preallocate a 100,000-byte
part, write 1,001 bytes, then call `written_prefix()`: it returns **4,096**.
Resuming skips bytes 1,001–4,095 and leaves zeros in the checkpoint. The final
size check cannot detect this because the parts were preallocated.

Fix: record exact durable offsets in a sidecar manifest, or use ordinary
append-only part files and resume from their lengths. Verify a trusted digest
before publishing the final checkpoint. Store URL/revision, size, and range
layout so parts cannot be reused for a different resource or connection count.

### 2. [P1] Range requests accept whole-file responses as valid parts

`scripts/fetch.py:58–73`

The downloader accepts HTTP 200 for every range and never checks
`Content-Range`. If a server ignores `Range`, a nonzero range receives bytes
from the beginning of the resource. Reproduction with a mocked response:
request bytes 5–9 of `0123456789`, receive HTTP 200, and the completed part is
`01234`, instead of `56789`. The final byte count still passes.

Fix: require HTTP 206 and the exact requested range and total size. A whole-file
HTTP 200 response should only be accepted by a separate single-stream path.
Treat premature EOF as failure and bound retries, including permanent errors.

### 3. [P1] The advertised empty-checkout setup cannot complete

`setup.sh:30–39`; `scripts/fetch.py:113–117`

`models/` is ignored by Git. Setup downloads only `model.safetensors`, without
creating its parent directory. The downloader consequently fails opening its
first part file. Even after creating the directory, preparation needs
`vocab.json`, `added_tokens.json`, `merges.txt`, and `normalizer.json`; conversion
also needs model, generation, and preprocessor configs. Setup fetches none of
these. Finally, verification requires an ignored WAV fixture that setup does
not supply.

Fix: create the directories, download an explicit manifest of checkpoint and
tokenizer assets at one pinned revision, and provide or generate a reproducible
verification fixture. Check for `uv` and `ffmpeg` at startup.

### 4. [P2] DEVICE does not select the server's inference device

`run.sh:17–19`; `scripts/serve.py:60–62`;
`wlk/whisperlivekit/local_agreement/backends.py:137–145`

The launcher uses DEVICE to select weights and coalescing, but the tuned backend
inherits a loader that always uses `device="auto", compute_type="auto"`.
A mocked model constructor confirms those exact kwargs. On a CUDA machine,
DEVICE=cpu can still select CUDA. DEVICE=gpu on a CPU host silently converts the
fp16 weights to float32 and runs on CPU, with GPU-oriented coalescing settings.
`metal` is accepted by the launcher but still launches the CTranslate2 backend.

Fix: normalize and validate DEVICE, pass explicit device and compute type to
the model loader, fail clearly when a requested accelerator is unavailable,
and reject Metal until an appropriate backend/artifact is supported.

### 5. [P2] Environment setup is tied to this absolute checkout path

`env.sh:3–11`

PROJECT_ROOT is hardcoded to `/home/ameer/Hobbies/livebuzzasr`. Moving or cloning
the project changes neither the cache locations nor the venv added to PATH.
On the GPU host described in the migration guide this may cause permission
errors or select another checkout's Python and dependencies.

Fix: derive the directory from `${BASH_SOURCE[0]}` when sourcing the file,
optionally allowing an explicit caller override.

### 6. [P2] Canonical diarization comparison drops subsequent chunks

`scripts/compare_diar_drivers.py:40–57`

`total` is initialized empty and passed to every `forward_streaming_step()`;
the returned `total_preds` is never assigned back to it. However, `prev_frames`
assumes the predictions accumulate. Once two consecutive chunks return equal
lengths, slicing at `prev_frames` discards the entire later prediction.
A mocked cumulative-output model fed three seconds emits only the first
second. NeMo's installed implementation documents and uses cumulative output.

Fix: retain the returned prediction history, and take the delta using its
previous length. Rerun comparisons before relying on the published canonical
versus WLK scores. Also move input tensors to the model device, handle the final
partial chunk, resample or reject non-16-kHz files, and flush both paths equally.
The current forced argmax also assigns a speaker during silence, which biases
false-alarm comparisons against a thresholded implementation.

### 7. [P2] Explicit API prompts are accepted but silently discarded

`scripts/serve.py:69, 74–78`;
`wlk/whisperlivekit/session_asr_proxy.py:165–178`

The upstream API advertises prompt/context support for faster-whisper. Its
session proxy merges explicit client context with rolling history. The tuned
backend's default `use_prompt=False` discards that combined value entirely.
A mock session with Persian terminology context forwards `initial_prompt=None`.
REST prompt and WebSocket context therefore appear supported but have no effect.

Fix: distinguish stable user terminology from rolling transcript history.
Preserve explicit context while disabling history, or accurately advertise and
reject unsupported context instead of silently ignoring it.

### 8. [P2] Verification can report success on failed transcription

`scripts/verify.sh:43–52`; `scripts/stream_test.py:50–79`

REST curl lacks `--fail`; a JSON error response is parsed successfully and its
missing text/word fields are printed as empty values. The WebSocket client
returns normally on ConnectionClosed (including error closes), ignores error
messages, and merely prints a timeout after 180 seconds. Neither path requires
a nonempty transcript or successful finalization. `verify: OK` is therefore
not a reliable acceptance result. The WebSocket wait budget also disagrees
with the 900-second REST budget used for slow CPU execution.

Fix: fail on HTTP errors and protocol error messages, require ready_to_stop,
raise on timeout and premature closure, and assert expected text/timestamps for
a known fixture. Allow the finish timeout to be configured.

### 9. [P2] Conversion deletes the working artifact before a replacement exists

`scripts/build_ct2.py:90–103`

`convert()` removes the destination recursively before launching the converter.
A converter failure, out-of-memory event, interrupted process, or full disk
destroys the previously usable model. The launcher only checks whether the
directory exists, so a partial rebuild can also pass its readiness check.

Fix: convert into a temporary sibling directory, patch and validate it, then
replace the published artifact with rollback protection. Check the required
artifact files or a completion marker before launching the server.

### 10. [P2] Offline verification and benchmarks use different decoding

`scripts/transcribe.py:24, 33–35`;
`scripts/bench_cpu.py:29–31`; `scripts/bench_encoder.py:39–42`

The server defaults to beam 1, temperature 0, repetition controls, and no history
conditioning. Offline transcription defaults to beam 5, the temperature fallback
list, and history conditioning; the benchmark scripts also omit several tuned
settings. The verification script compares paths with different behavior.
These results cannot establish server quality or isolate streaming overhead,
and offline runs can reproduce the expensive fallback behavior the launcher
was introduced to avoid.

Fix: centralize the decoding profile and expose deliberate baseline/tuned
profiles in offline and benchmark tools. Print the resolved profile with each
result and compare the same profile across offline, REST, and streaming.

## Optimization and maintainability opportunities

- **Make builds reproducible.** Setup clones upstream HEAD and installs
  unconstrained dependencies. The adaptation patches upstream module globals
  and depends on its internal class contracts. Pin an upstream commit, model
  revision, and dependency set; add adaptation-level contract tests. The nested
  `wlk/uv.lock` is not used by the outer setup script.
- **Reduce conversion validation memory.** Despite its metadata-only comment,
  `build_ct2.validate()` constructs the full fp16 model on CPU. A local check
  automatically converted it to float32 and took 61.5 seconds. The int8 check
  also keeps one CT2 model alive while constructing a second faster-whisper
  model. Validate JSON/tokenizer metadata without loading weights, then load
  one runtime instance per required device and release it promptly.
- **Expose CPU thread tuning.** The server's inherited loader exposes neither
  `cpu_threads` nor worker count through the adaptation. Benchmark thread counts
  with the actual tuned decoding profile and representative simultaneous
  sessions, then provide configuration. Avoid assuming all logical cores are
  optimal on this laptop.
- **Measure streaming quality and latency separately.** Record normalized and
  raw Persian CER/WER, time to first partial, time to committed words, final
  latency, ASR call count, and peak memory. Include silence, noise, short final
  utterances, numbers, Persian/Latin mixtures, ZWNJ, and Arabic/Persian kaf/yeh.
  Keep display spelling separate from evaluation normalization.
- **Validate alignment empirically.** Reusing large-v3 alignment heads is a
  plausible starting point, but fine-tuning can change attention. Noncollapsed
  timestamps alone do not establish word-boundary accuracy. Check against a
  manually aligned Persian fixture before treating the heads as validated.
- **Add Persian text direction to transcript content.** The web renderer emits
  `.textcontent` without `dir="auto"` or direction isolation. Use direction-aware
  transcript spans and isolate timestamps and speaker labels for mixed text.
- **Strengthen diarization evaluation boundaries.** `eval_diar` processes at
  most one model chunk per call and pads only once at EOF. Large values of
  `--chunk` can leave unscored audio buffered. Drain all full chunks and flush
  the tail explicitly. Report the treatment of silence, overlap, and boundary
  collars alongside DER; current labels represent only one speaker per frame.

## Validation and limits

- Shell syntax and AST parsing passed for all outer scripts.
- Tested the downloader's allocation-based resume offset and acceptance of
  incorrect HTTP 200 range responses with local, deterministic reproductions.
- Reproduced prompt loss, automatic device kwargs, and canonical diarization
  truncation using mocks, without running large-model inference.
- Full tokenizer regeneration into `/tmp` passed: all 13 checked special-token
  IDs matched, the Persian sample round-tripped exactly, and Transformers
  5.20.2 loaded all 53,474 tokens.
- Focused upstream tests passed: **14/14**, from `test_asr_coalescing.py` and
  `test_session_context.py`.
- The broader upstream test selection reached 30 passing tests before stalling
  and was interrupted. A narrower selection including coalescing boundary and
  server tests also stalled after three passes and was interrupted. These runs
  do not establish a complete test-suite pass or the cause of the stalls.
- No CUDA device was available. GPU performance and migration instructions were
  not runtime-validated. No new end-to-end ASR accuracy benchmark was performed.
- Existing user edits were preserved. Test dependencies were extracted from
  existing wheels under `/tmp`; the project venv was not modified.
