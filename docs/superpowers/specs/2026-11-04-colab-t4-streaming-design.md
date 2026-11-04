# Colab T4 live-streaming demo — design

Status: awaiting approval
Date: 2026-11-04

## Intent

Run the existing WhisperLiveKit streaming stack against the private
`ameerhossein/liquidwhisper-neyshekar-ct2-gpu` CTranslate2 model on a free
Colab T4, and expose the live microphone page through a Cloudflare quick
tunnel. The local machine has no GPU and a slow network; Colab does the model
download and inference, while the browser only carries the microphone stream.

## Goals

- Deliver `notebooks/colab_t4_streaming.ipynb`, runnable top-to-bottom on a
  Colab T4 runtime.
- Clone `https://github.com/ameerhossein/LiquidWhisperLiveKit.git`
  (`origin/main`, currently `0d01eb3`).
- Use the exact repo runtime: `setup.sh` creates a Python 3.12 `.venv`,
  installs the pinned CUDA 12.9 torch pair and `requirements.lock`, and clones
  WhisperLiveKit at `363e4f6`.
- Read the HF token from the Colab Secret `HF_TOKEN`; never write it into the
  notebook. Default model repo:
  `ameerhossein/liquidwhisper-neyshekar-ct2-gpu`, with optional
  `HF_REVISION` and `HF_SUBFOLDER`.
- Auto-detect the CT2 directory when `HF_SUBFOLDER` is blank: list repo files,
  find `model.bin`, and use its parent only if the other required CT2 files
  are present. Fail with an actionable message otherwise.
- Start the server on `0.0.0.0:8000` with `DEVICE=gpu`, `--pcm-input`, and
  `ASR_COALESCE=0`.
- Expose decoding as Colab form parameters: `WLK_BEAM` (default `1`),
  `WLK_TEMPERATURE` (default `0.0`), and `ASR_COALESCE` (default `0`). Changing
  them and re-running the start-server cell applies them.
- Start `cloudflared` quick tunnel, wait for `/health`, parse and print the
  `https://*.trycloudflare.com` URL.
- Include an optional self-test cell that streams `testdata/verify-fa.wav`
  through `scripts/stream_test.py` and prints partial updates.
- Include GPU, model-layout, CTranslate2/CUDA, and cuDNN-9 preflight checks
  with explicit remediation.
- Add a short README section with an “Open in Colab” link.

## Non-goals

- No changes to WhisperLiveKit’s HTML/JS and no beam-size control in the live
  UI. The shared ASR instance would need per-session decoding plumbing, which
  is a separate feature.
- No model conversion, training, or tokenizer work.
- No diarization in the first version.
- No persistent/public deployment, authentication layer, or uptime guarantee
  beyond the temporary quick tunnel.
- No changes to the local CPU path.

## Anti-goals

- Baking an HF token into a notebook cell or committed file.
- Silently falling back to CPU, int8, or a default model when CUDA or the
  private model is unavailable.
- Installing a loose, unpinned dependency set into Colab’s system Python.
- Claiming the demo works without `/health` and at least one streaming update.

## Constraints

- Colab T4: ~16 GB VRAM, Python 3.12, CUDA 12.x, cuDNN 9 required by
  CTranslate2.
- The model repo is private; the token needs read access.
- `wlk/` is gitignored in this repo, so `setup.sh` must clone it at the pinned
  commit.
- The CT2 artifact must contain `model.bin`, `config.json`,
  `vocabulary.json`, `tokenizer.json`, and `preprocessor_config.json`.
- Quick tunnels get a random URL, are publicly reachable while running, and
  die with the Colab session.
- Colab disk is ephemeral; each fresh session re-downloads the runtime and
  model. Caching on Drive is intentionally deferred.

## Approaches considered

### A — Exact repo runtime via `setup.sh` (chosen)

Clone the repo, set `HF_REPO_ID`, run `DEVICE=gpu ./setup.sh`, then
`run.sh` plus `cloudflared`.

Trade-off: a few extra minutes and several GB of Colab disk, but it preserves
the pinned Python, torch, CTranslate2, and WhisperLiveKit versions that the
repo’s CPU results were validated against. On Colab’s network the extra
download is cheap.

### B — Lean Colab-native install (rejected)

Reuse Colab’s Python/torch, `pip install` the pinned requirements, download
the model, and run `scripts/serve.py`.

Trade-off: faster first cell, but Colab’s preinstalled numpy/torch/CUDA stack
drifts from `requirements.lock`, inviting the exact CUDA and dependency
surprises the repo’s setup exists to avoid.

### Beam control in the UI (rejected for v1)

WLK shares one ASR backend across sessions; the session proxy isolates
language and prompt context only. A dropdown would require per-session
decoding overrides, lock-safe application, and custom UI/JS. A notebook form
parameter gives the same experiment with no upstream changes.

## Notebook flow

1. Markdown: runtime selection, Colab Secret setup, what the tunnel exposes.
2. Config cell with `#@param` fields: repo URL/ref, HF repo, revision,
   subfolder, beam, temperature, coalesce, port.
3. GPU check: `nvidia-smi`, compute capability, free VRAM/disk.
4. Token and private-repo access check via `huggingface_hub`.
5. Clone or update the project.
6. Run `DEVICE=gpu HF_REPO_ID=... ./setup.sh`; persist the resolved model path.
7. CTranslate2 CUDA preflight, cuDNN-9 remediation if needed, model-file
   validation.
8. Start the server in the background; poll `/health`; show the log tail on
   failure.
9. Download/start `cloudflared`; parse and print the public URL.
10. Optional WAV self-test against `ws://127.0.0.1:8000/asr`.
11. Logs, stop-server, and stop-tunnel cells.

## Decisions made

- `--pcm-input` is always on so the microphone UI (audio worklet) and
  `stream_test.py` share one server mode.
- `HOST=0.0.0.0` is used only for the tunnel-facing server.
- `WLK_BEAM=1` remains the default based on the repo’s measured repetition
  behavior; the T4 experiment can raise it.
- The notebook fails closed on missing GPU, missing token, inaccessible repo,
  incomplete CT2 files, or unsupported CUDA compute types.
- `LD_LIBRARY_PATH` will include the venv’s NVIDIA cuDNN/cuBLAS library
  directories so CTranslate2 can find `libcudnn_ops.so.9`; if the libraries
  are absent, the notebook installs `nvidia-cudnn-cu12==9.*`.
- The quick-tunnel URL is printed as the final output, not stored in a file
  that could be mistaken for a durable endpoint.

## Verification

Local (no GPU):

- Validate notebook JSON with `nbformat`.
- Syntax-check extracted shell cells with `bash -n`.
- `git diff --check`.

In Colab:

- GPU cell reports Tesla T4 and CUDA available.
- CTranslate2 reports `float16` in supported CUDA compute types.
- Model validation reports all five required files.
- `/health` returns OK.
- Tunnel URL opens the WhisperLiveKit page over HTTPS; microphone permission
  works and partial Persian text appears.
- The WAV self-test prints at least one partial update and `ready_to_stop`.

## Left to the builder

- Exact cell code, Cloudflare log parsing, health polling, and error messages.
- Whether a Drive-based cache is worth adding after the first real run.
- Choosing the README placement and Colab badge URL.
