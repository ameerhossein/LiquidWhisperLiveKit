#!/usr/bin/env bash
# One-shot setup from an empty checkout. Needs ~10 GB free disk.
#   1. Python 3.12 venv + whisperlivekit (offline from the copied uv cache)
#   2. fix the BuzzASR tokenizer
#   3. download the 2.9 GiB checkpoint (resumable)
#   4. convert to CTranslate2 (int8 for CPU, float16 for CUDA)
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

for tool in uv git ffmpeg curl; do
  command -v "$tool" >/dev/null || { echo "Required tool is missing: $tool" >&2; exit 2; }
done
export DEVICE="${DEVICE:-cpu}"
case "$DEVICE" in
  cpu) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
  gpu|cuda) TORCH_INDEX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}" ;;
  *) echo "DEVICE must be cpu, gpu, or cuda" >&2; exit 2 ;;
esac
WLK_REVISION="363e4f6d029694d9c81ae548beddd9d3c88a3637"
if [[ ! -e wlk/.git ]]; then
  git init wlk
  git -C wlk remote add origin https://github.com/QUENTINFUXA/WHISPERLIVEKIT.git
fi
if ! git -C wlk rev-parse --verify HEAD >/dev/null 2>&1; then
  git -C wlk fetch --depth 1 origin "$WLK_REVISION"
  git -C wlk checkout --detach "$WLK_REVISION"
fi
if [[ "$(git -C wlk rev-parse HEAD)" != "$WLK_REVISION" ]]; then
  echo "wlk must be at $WLK_REVISION; existing checkout left untouched" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Setup requires a Python 3.12 venv"'
# Keep torch and torchaudio on the same release (required by their native ABI).
if [[ -d wheels && "$DEVICE" == cpu ]]; then
  uv pip install --no-index --find-links wheels --python .venv/bin/python \
    torch==2.11.0 torchaudio==2.11.0 -r requirements.lock -e ./wlk
else
  uv pip install --python .venv/bin/python torch==2.11.0 torchaudio==2.11.0 \
    --index-url "$TORCH_INDEX"
  uv pip install --python .venv/bin/python -r requirements.lock -e ./wlk
fi
.venv/bin/python scripts/fetch_model.py
.venv/bin/python scripts/prepare_tokenizer.py
.venv/bin/python scripts/build_ct2.py --validate-runtime --alignment-heads-fallback large-v3
./scripts/verify.sh
