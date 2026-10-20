#!/usr/bin/env bash
# Run WhisperLiveKit with BuzzASR/persian.
#
#   DEVICE=cpu  ./run.sh          # 127.0.0.1:8000, opens Web UI
#   DEVICE=gpu  ./run.sh          # same, on a CUDA machine
#   PORT=9000 HOST=0.0.0.0 ./run.sh
#   ./run.sh --min-chunk-size 0.5 # extra flags go straight to `wlk serve`
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

DEVICE="${DEVICE:-cpu}"
case "$DEVICE" in
  cpu)            MODEL_DIR="models/buzzasr-persian-ct2-int8" ;;
  gpu|cuda|metal) MODEL_DIR="models/buzzasr-persian-ct2-fp16" ;;
  *) echo "DEVICE must be cpu or gpu (got '$DEVICE')" >&2; exit 2 ;;
esac

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model not built yet: $MODEL_DIR" >&2
  echo "Run: source env.sh && .venv/bin/python scripts/build_ct2.py" >&2
  exit 2
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# On CPU every ASR call costs a full 30 s encoder pass (~10-12 s on a laptop),
# so defer ASR until this much new audio has accrued: fewer, larger passes.
# Set ASR_COALESCE=0 for lowest latency if you have a GPU.
COALESCE="${ASR_COALESCE:-0}"
[[ "$DEVICE" == "cpu" ]] && COALESCE="${ASR_COALESCE:-6}"
# WhisperLiveKit's default warmup downloads a 60 kB JFK clip from GitHub.
# Prefer a local Persian clip so startup works offline ("" would skip warmup).
WARMUP="${WARMUP_FILE:-testdata/warmup-fa.wav}"
[[ -f "$WARMUP" ]] || WARMUP=""

echo "BuzzASR/persian on WhisperLiveKit — device=$DEVICE model=$MODEL_DIR"
# scripts/serve.py = wlk + CPU-sane decoding (no temperature-fallback retries)
exec .venv/bin/python scripts/serve.py serve \
  --backend faster-whisper \
  --backend-policy localagreement \
  --model_dir "$MODEL_DIR" \
  --language fa \
  --warmup-file "$WARMUP" \
  --asr-coalesce-min-s "$COALESCE" \
  --rest-timeout "${REST_TIMEOUT:-900}" \
  --log-level "${LOG_LEVEL:-INFO}" \
  --host "$HOST" --port "$PORT" \
  "$@"
