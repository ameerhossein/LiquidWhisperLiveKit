#!/usr/bin/env bash
# Install the inference runtime. Supply a prepared CT2 model locally or on HF.
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
MODEL_DIR="$(.venv/bin/python scripts/settings.py --resolve-model)"
.venv/bin/python scripts/settings.py --check-model "$MODEL_DIR"
echo "Inference runtime ready. Model: $MODEL_DIR"
echo "Run ./run.sh to serve; ./scripts/verify.sh runs acceptance checks."
