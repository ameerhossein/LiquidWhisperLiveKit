#!/usr/bin/env bash
# Fail-closed offline + REST + WebSocket acceptance check.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
export DEVICE="${DEVICE:-cpu}"
case "$DEVICE" in
  cpu) DEFAULT_MODEL_DIR="models/buzzasr-persian-ct2-int8"; DEFAULT_COALESCE=6 ;;
  gpu|cuda) DEFAULT_MODEL_DIR="models/buzzasr-persian-ct2-fp16"; DEFAULT_COALESCE=0 ;;
  *) echo "DEVICE must be cpu, gpu, or cuda (Metal is unsupported)" >&2; exit 2 ;;
esac
export MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
.venv/bin/python scripts/settings.py --check-model "$MODEL_DIR"
AUDIO="${1:-testdata/verify-fa.wav}"
EXPECTED="${EXPECTED_FILE:-}"
if [[ "$AUDIO" == testdata/verify-fa.wav ]]; then EXPECTED="${EXPECTED:-testdata/verify-fa.txt}"; fi
PORT="${VERIFY_PORT:-8791}"
TIMEOUT="${REST_TIMEOUT:-900}"
[[ -f "$AUDIO" ]] || { echo "no such audio: $AUDIO" >&2; exit 2; }
mkdir -p .cache
RESULTS_DIR="$(mktemp -d .cache/verify-XXXXXX)"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
check_result() {
  local result_file="$1"
  shift
  if [[ -n "$EXPECTED" ]]; then
    .venv/bin/python scripts/validate_result.py "$result_file" --expected-file "$EXPECTED" "$@"
  else
    .venv/bin/python scripts/validate_result.py "$result_file" "$@"
  fi
}
echo "== 1/4 offline transcription"
.venv/bin/python scripts/transcribe.py "$AUDIO" --model "$MODEL_DIR" --words \
  --json-output "$RESULTS_DIR/offline.json"
check_result "$RESULTS_DIR/offline.json" --words

echo "== 2/4 server startup on 127.0.0.1:$PORT"
.venv/bin/python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(sys.argv[1])))
PY
.venv/bin/python scripts/serve.py serve --backend faster-whisper --backend-policy localagreement \
  --model_dir "$MODEL_DIR" --language fa --host 127.0.0.1 --port "$PORT" \
  --pcm-input --log-level "${VERIFY_LOG_LEVEL:-INFO}" --asr-coalesce-min-s "${ASR_COALESCE:-$DEFAULT_COALESCE}" \
  --rest-timeout "$TIMEOUT" --warmup-file "" > "$RESULTS_DIR/server.log" 2>&1 &
SERVER_PID=$!
for ((i=0; i<300; i++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || { cat "$RESULTS_DIR/server.log"; exit 1; }
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" > "$RESULTS_DIR/health.json" 2>/dev/null; then break; fi
  sleep 1
done
# Catch startup failure/port collisions instead of accepting another server's health.
kill -0 "$SERVER_PID"
.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("ready") is True' "$RESULTS_DIR/health.json"
echo "== 3/4 REST transcription"
curl --fail-with-body -sS --dump-header "$RESULTS_DIR/rest.headers" --max-time "$TIMEOUT" "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
  -F "file=@$AUDIO" -F response_format=verbose_json > "$RESULTS_DIR/rest.json"
check_result "$RESULTS_DIR/rest.json" --words

echo "== 4/4 WebSocket transcription"
.venv/bin/python scripts/stream_test.py "$AUDIO" --url "ws://127.0.0.1:$PORT/asr" \
  --realtime 1 --timeout "$TIMEOUT" --require-text --json-output "$RESULTS_DIR/stream.json"
check_result "$RESULTS_DIR/stream.json"
echo "verify: OK (results: $RESULTS_DIR)"
