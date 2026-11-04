# Colab T4 Live Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-click Google Colab T4 notebook that runs this repo's WhisperLiveKit streaming stack against the private `ameerhossein/liquidwhisper-neyshekar-ct2-gpu` model and exposes the microphone page through a temporary Cloudflare tunnel.

**Architecture:** The notebook clones this repository, uses the existing `setup.sh` to build the pinned Python 3.12/CUDA runtime and download the private CT2 model, runs `scripts/serve.py` via `run.sh`, and starts `cloudflared` as a quick tunnel. Small pure helpers for CT2-layout detection and tunnel-URL parsing live in `scripts/colab_support.py` so they can be tested without a GPU.

**Tech Stack:** Python 3.12, `uv`, PyTorch CUDA 12.9, CTranslate2, faster-whisper, WhisperLiveKit, Cloudflare quick tunnels, Jupyter/Colab.

**Spec:** `docs/superpowers/specs/2026-11-04-colab-t4-streaming-design.md`

## Global Constraints

- Use the exact repo runtime: `DEVICE=gpu ./setup.sh`, which creates `.venv` with Python 3.12, installs `torch==2.11.0`/`torchaudio==2.11.0` from the cu129 index, installs `requirements.lock`, and clones WhisperLiveKit at `363e4f6d029694d9c81ae548beddd9d3c88a3637`.
- Never install the project runtime into Colab's system Python. The only system-Python package the notebook installs is `uv`.
- Read the HF token from the Colab Secret `HF_TOKEN`; never commit or print a token literal.
- Default model repo: `ameerhossein/liquidwhisper-neyshekar-ct2-gpu`; support optional `HF_REVISION` and `HF_SUBFOLDER`.
- A valid CT2 artifact has exactly these files: `model.bin`, `config.json`, `vocabulary.json`, `tokenizer.json`, `preprocessor_config.json`.
- Start the server with `DEVICE=gpu`, `HOST=0.0.0.0`, `PORT=8000`, `--pcm-input`, `ASR_COALESCE=0`, `WLK_BEAM=1`, `WLK_TEMPERATURE=0.0`.
- Do not modify WhisperLiveKit's HTML/JS, add a beam dropdown, or change the local CPU path.
- The quick-tunnel URL is public while the notebook runs; include a warning and a stop cell.
- Fail closed on missing GPU, invalid token, inaccessible repo, incomplete CT2 layout, or unsupported CUDA compute types.

## Review Focus

- **Missing GPU or CPU runtime selected:** the notebook must stop before installing or downloading, with the exact runtime-switch instruction.
- **Missing/invalid HF token or inaccessible private repo:** the notebook must stop with a 401/403/404 explanation and never echo the token.
- **Model repo has a nested, incomplete, or ambiguous CT2 directory:** auto-detect only one complete directory; otherwise name the candidates and require `HF_SUBFOLDER`.
- **CTranslate2 cannot find cuDNN 9 or reports no `float16`:** install `nvidia-cudnn-cu12==9.*` into the venv, retry once, and fail with the captured error if it still fails.
- **Server exits or tunnel URL never appears:** show the relevant log tail and do not print a URL that was not observed.

---

### Task 1: Pure Colab helpers

**Files:**
- Create: `scripts/colab_support.py`
- Test: `tests/test_colab_support.py`

**Interfaces:**
- Consumes: nothing (standard library only).
- Produces:
  - `REQUIRED_CT2_FILES: tuple[str, ...]`
  - `find_ct2_subfolder(files: Iterable[str], requested: str = "") -> str`
  - `parse_tunnel_url(text: str) -> str | None`
  - `nvidia_library_dirs(site_packages: str | Path) -> list[str]`
  - `find_ct2_subfolder` returns `""` for the repo root and raises `ValueError` with `No complete CTranslate2` or `Multiple complete CTranslate2` in the message.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_colab_support.py
from pathlib import Path

import pytest

import colab_support
import settings

REQUIRED = list(colab_support.REQUIRED_CT2_FILES)


def files_at(prefix: str) -> list[str]:
    return [f"{prefix}{name}" for name in REQUIRED]


def test_required_files_match_runtime_validation():
    assert set(colab_support.REQUIRED_CT2_FILES) == set(settings.REQUIRED_MODEL_FILES)


def test_find_root_ct2():
    assert colab_support.find_ct2_subfolder(files_at("")) == ""


def test_find_nested_ct2():
    assert colab_support.find_ct2_subfolder(files_at("runs/a/ct2/")) == "runs/a/ct2"


def test_preferred_subfolder_is_validated():
    files = files_at("a/") + files_at("b/")
    assert colab_support.find_ct2_subfolder(files, "b") == "b"


def test_incomplete_ct2_is_rejected():
    with pytest.raises(ValueError, match="No complete CTranslate2"):
        colab_support.find_ct2_subfolder(["model.bin", "config.json"])


def test_ambiguous_ct2_is_rejected():
    with pytest.raises(ValueError, match="Multiple complete CTranslate2"):
        colab_support.find_ct2_subfolder(files_at("a/") + files_at("b/"))


def test_requested_subfolder_must_be_complete():
    with pytest.raises(ValueError, match="not a complete CT2 directory"):
        colab_support.find_ct2_subfolder(files_at("a/"), "b")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("visit https://one-two.trycloudflare.com now", "https://one-two.trycloudflare.com"),
        ("https://a.trycloudflare.com\nhttps://b.trycloudflare.com", "https://a.trycloudflare.com"),
        ("not ready", None),
    ],
)
def test_parse_tunnel_url(text, expected):
    assert colab_support.parse_tunnel_url(text) == expected


def test_nvidia_library_dirs_only_returns_existing_dirs(tmp_path):
    for relative in ("nvidia/cudnn/lib", "nvidia/cublas/lib", "nvidia/cuda_runtime/lib"):
        (tmp_path / relative).mkdir(parents=True)
    assert colab_support.nvidia_library_dirs(tmp_path) == [
        str(tmp_path / "nvidia/cudnn/lib"),
        str(tmp_path / "nvidia/cublas/lib"),
        str(tmp_path / "nvidia/cuda_runtime/lib"),
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_colab_support.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'colab_support'`.

- [ ] **Step 3: Implement `scripts/colab_support.py`**

Use `re` for the tunnel URL and `collections.defaultdict` for grouping paths by parent. `find_ct2_subfolder` must strip `/` from `requested`, return `requested` only when its directory has all required files, and otherwise list complete candidates or directories containing `model.bin` in the error. `parse_tunnel_url` returns the first `https://<label>.trycloudflare.com` match. `nvidia_library_dirs` returns existing `nvidia/{cudnn,cublas,cuda_runtime}/lib` directories in that order.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_colab_support.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/colab_support.py tests/test_colab_support.py
git commit -m "feat: add tested Colab helpers"
```

---

### Task 2: Colab notebook and notebook contract tests

**Files:**
- Create: `notebooks/colab_t4_streaming.ipynb`
- Test: `tests/test_colab_notebook.py`

**Interfaces:**
- Consumes: `scripts/colab_support.py` via `sys.path.insert(0, str(PROJECT_DIR / "scripts"))`.
- Produces: a JSON notebook with cells in this order:
  1. Markdown intro and public-tunnel warning.
  2. `#@param` configuration: repo URL/ref, HF repo/revision/subfolder, `WLK_BEAM=1`, `WLK_TEMPERATURE="0.0"`, `ASR_COALESCE="0"`, `PORT=8000`.
  3. GPU/disk preflight.
  4. Clone/update `PROJECT_DIR=/content/LiquidWhisperLiveKit`.
  5. `HF_TOKEN` Secret + repo access + `find_ct2_subfolder`.
  6. `uv` install + `DEVICE=gpu HF_REPO_ID=... ./setup.sh`; persist resolved model path to `/content/livebuzzasr-colab/model-dir.txt`.
  7. CTranslate2/CUDA/cuDNN preflight + `settings.py --check-model`.
  8. Start `run.sh --pcm-input`; wait for `/health` with `ready: true`.
  9. Download/start `cloudflared`; parse URL with `parse_tunnel_url`; verify `/health` through the tunnel.
  10. Self-test `scripts/stream_test.py testdata/verify-fa.wav --require-text`.
  11. Stop server/tunnel and print log tails.
  12. Markdown troubleshooting.

- [ ] **Step 1: Write the failing notebook contract tests**

```python
# tests/test_colab_notebook.py
import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_t4_streaming.ipynb"


def notebook():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert data["cells"]
    return data


def code(data):
    return "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")


def text(data):
    return "\n".join("".join(cell["source"]) for cell in data["cells"])


def test_notebook_exists_and_is_valid_json():
    assert NOTEBOOK.is_file()
    notebook()


def test_config_defaults():
    source = code(notebook())
    assert 'HF_REPO_ID = "ameerhossein/liquidwhisper-neyshekar-ct2-gpu"' in source
    assert "WLK_BEAM = 1" in source
    assert 'WLK_TEMPERATURE = "0.0"' in source
    assert 'ASR_COALESCE = "0"' in source
    assert "PORT = 8000" in source


def test_uses_secret_and_tested_helpers():
    source = code(notebook())
    assert 'userdata.get("HF_TOKEN")' in source
    assert "find_ct2_subfolder(files, HF_SUBFOLDER)" in source
    assert "parse_tunnel_url(" in source


def test_runtime_contract():
    source = code(notebook())
    assert "nvidia-smi" in source
    assert "No CUDA GPU detected" in source
    assert "T4 GPU" in source
    assert "401" in source and "403" in source and "404" in source
    assert '"HOST": "0.0.0.0"' in source
    assert '"DEVICE": "gpu"' in source
    assert '"--pcm-input"' in source
    assert '"ASR_COALESCE": str(ASR_COALESCE)' in source
    assert '"nvidia-cudnn-cu12==9.*"' in source
    assert 'get_supported_compute_types("cuda")' in source
    assert "float16" in source
    assert "/health" in source
    assert '"ready"' in source
    assert '"--no-autoupdate"' in source
    assert "trycloudflare.com" in source
    assert "testdata/verify-fa.wav" in source


def test_public_tunnel_is_warned_about():
    assert "public while" in text(notebook()).lower()


def test_no_hf_token_literal():
    source = text(notebook())
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_colab_notebook.py -q`
Expected: FAIL with `AssertionError` because the notebook does not exist.

- [ ] **Step 3: Create the notebook**

Build valid `nbformat` 4 JSON. Keep all long-running commands as Python `subprocess` calls so their output streams into the cell. The configuration cell must define `PROJECT_DIR`, `STATE_DIR`, `MODEL_DIR_FILE`, `SERVER_LOG`, `TUNNEL_LOG`, `SERVER_PID`, `TUNNEL_PID`, and `CLOUDFLARED` exactly as named below.

Required behavior in the cells:

- GPU cell rejects `not torch.cuda.is_available()`, compute capability `< 7.0`, and free disk `< 12 GiB`; prints the GPU name and VRAM.
- Clone cell uses `git clone --depth 1 --branch REPO_REF` on first run, otherwise `git fetch --depth 1 origin REPO_REF` + `git checkout --force FETCH_HEAD`; `git clean -ffdx -e .venv -e .cache -e models` removes stale checkouts (including the nested `wlk` clone) while preserving the venv, HF cache, and local models.
- Auth cell accepts `HF_TOKEN` from `google.colab.userdata` then from the environment, and only then prompts with `getpass`. It calls `GET https://huggingface.co/api/models/{HF_REPO_ID}` and `GET .../tree/{revision}?recursive=true` with a Bearer token, maps 401/403/404 to actionable `SystemExit` messages, and calls `find_ct2_subfolder(files, HF_SUBFOLDER)`.
- Setup cell installs `uv` into system Python, runs `bash setup.sh` with `DEVICE=gpu`, `HF_REPO_ID`, `HF_REVISION`, `HF_SUBFOLDER`, and `HF_TOKEN`, then resolves and persists the model directory with `.venv/bin/python scripts/settings.py --resolve-model`.
- Preflight cell sets `LD_LIBRARY_PATH` from `nvidia_library_dirs(site_packages)`, runs a `.venv` Python snippet that asserts `torch.cuda.is_available()` and `"float16" in ctranslate2.get_supported_compute_types("cuda")`, retries once after `uv pip install --python .venv/bin/python "nvidia-cudnn-cu12==9.*"` if stderr contains `cudnn`, then runs `scripts/settings.py --check-model`.
- Server cell removes `HF_REPO_ID`/`HF_REVISION`/`HF_SUBFOLDER` from the environment, sets `MODEL_DIR` from the persisted file, starts `["bash", "run.sh", "--pcm-input"]` with `start_new_session=True`, writes `SERVER_PID`, polls `http://127.0.0.1:{PORT}/health` for up to 240 s until `ready` is true, and prints the log tail on failure.
- Tunnel cell downloads `https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64` to `/content/bin/cloudflared` with `curl -L --fail --retry 3`, starts `cloudflared tunnel --url http://127.0.0.1:{PORT} --no-autoupdate`, polls the log for `parse_tunnel_url`, verifies `/health` through the public URL, and displays the URL as a link.
- Self-test cell runs `.venv/bin/python scripts/stream_test.py testdata/verify-fa.wav --url ws://127.0.0.1:{PORT}/asr --realtime 1.0 --timeout 300 --require-text`.
- Stop cell terminates both process groups with `SIGTERM`/`SIGKILL`, deletes the PID files, and prints the last 20 lines of each log.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_colab_notebook.py tests/test_colab_support.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/colab_t4_streaming.ipynb tests/test_colab_notebook.py
git commit -m "feat: add Colab T4 streaming notebook"
```

---

### Task 3: README entry point

**Files:**
- Modify: `README.md` after the Quick start section, before `## Decoding behavior`.

**Interfaces:**
- Consumes: the notebook path from Task 2.
- Produces: a `## Run on a Colab T4` section with a Colab badge pointing at
  `https://colab.research.google.com/github/ameerhossein/LiquidWhisperLiveKit/blob/main/notebooks/colab_t4_streaming.ipynb`.

- [ ] **Step 1: Add the README section**

Include:

- The Colab badge.
- Runtime → T4 GPU, `HF_TOKEN` Secret with read access to the private model repo, Run all.
- Open the printed `https://*.trycloudflare.com` URL and allow microphone access.
- A warning that the tunnel is public while running; run the stop cell when done.
- A note that `WLK_BEAM`, `WLK_TEMPERATURE`, and `ASR_COALESCE` are form parameters and the start-server/tunnel cells must be re-run after changing them.

- [ ] **Step 2: Verify the pointer**

Run: `rg -n "Run on a Colab T4|colab_t4_streaming.ipynb|colab-badge" README.md`
Expected: the heading, badge URL, and notebook path all match.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: link the Colab T4 streaming notebook"
```

---

### Task 4: Final verification

**Files:** none.

- [ ] **Step 1: Validate notebook JSON**

Run: `.venv/bin/python -m json.tool notebooks/colab_t4_streaming.ipynb > /dev/null`
Expected: exit 0.

- [ ] **Step 2: Run the targeted tests**

Run: `.venv/bin/python -m pytest tests/test_colab_support.py tests/test_colab_notebook.py -q`
Expected: PASS.

- [ ] **Step 3: Check the diff and secrets**

Run: `git diff --check && ! rg -n "hf_[A-Za-z0-9]{20,}" notebooks scripts tests README.md`
Expected: exit 0.

- [ ] **Step 4: Record Colab acceptance evidence**

After opening the notebook in Colab with a T4 runtime, confirm in the notebook output:

- GPU cell prints a Tesla T4 and CUDA available.
- Preflight prints `float16` in the CTranslate2 CUDA compute types.
- `/health` returns `ready: true`.
- The tunnel cell prints a `https://*.trycloudflare.com` URL and a successful tunnel health check.
- The self-test prints at least one partial update and `ready_to_stop`.
- The browser opens the URL and produces Persian partial text from the microphone.
