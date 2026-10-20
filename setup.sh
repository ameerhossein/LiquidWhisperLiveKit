#!/usr/bin/env bash
# One-shot setup from an empty checkout. Needs ~10 GB free disk.
#   1. Python 3.12 venv + whisperlivekit (offline from the copied uv cache)
#   2. fix the BuzzASR tokenizer
#   3. download the 2.9 GiB checkpoint (resumable)
#   4. convert to CTranslate2 (int8 for CPU, float16 for CUDA)
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

if [[ ! -d wlk/.git ]]; then
  git clone --depth 1 https://github.com/QUENTINFUXA/WHISPERLIVEKIT.git wlk
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
if [[ -d wheels ]]; then
  uv pip install --no-index --find-links wheels --python .venv/bin/python \
    torch torchaudio faster-whisper transformers safetensors
  uv pip install --no-index --find-links wheels --python .venv/bin/python -e ./wlk
else
  echo "no local wheels/ - installing from PyPI (slow on this link)" >&2
  uv pip install --python .venv/bin/python torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv/bin/python faster-whisper transformers safetensors
  uv pip install --python .venv/bin/python -e ./wlk
fi

if [[ ! -f models/buzzasr-persian/model.safetensors ]]; then
  echo "downloading BuzzASR/persian weights (resumable)..."
  python3 -u scripts/fetch.py \
    "https://huggingface.co/BuzzASR/persian/resolve/main/model.safetensors" \
    models/buzzasr-persian/model.safetensors --connections 8
fi

.venv/bin/python scripts/prepare_tokenizer.py
.venv/bin/python scripts/build_ct2.py
./scripts/verify.sh
