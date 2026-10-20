#!/usr/bin/env bash
# End-to-end check: offline transcription, REST API, and streaming WebSocket.
# Usage: ./scripts/verify.sh [audio.wav]
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

AUDIO="${1:-testdata/persian-10s.wav}"
PORT="${VERIFY_PORT:-8791}"
[[ -f "$AUDIO" ]] || { echo "no such audio: $AUDIO" >&2; exit 2; }
[[ -d models/buzzasr-persian-ct2-int8 ]] || {
  echo "missing models/buzzasr-persian-ct2-int8 - run scripts/build_ct2.py first" >&2; exit 2; }

echo "== 1/4 offline transcription (faster-whisper, int8)"
.venv/bin/python scripts/transcribe.py "$AUDIO" --words

echo
echo "== 2/4 starting server on 127.0.0.1:$PORT (pcm-input for the WS test)"
.venv/bin/python scripts/serve.py serve --backend faster-whisper --backend-policy localagreement \
  --model_dir models/buzzasr-persian-ct2-int8 --language fa \
  --host 127.0.0.1 --port "$PORT" --pcm-input --log-level WARNING \
  --asr-coalesce-min-s 6 --rest-timeout 900 \
  --warmup-file testdata/warmup-fa.wav \
  > .cache/verify-server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 300); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" || { echo "server failed to start"; tail -30 .cache/verify-server.log; exit 1; }
echo

echo "== 3/4 REST /v1/audio/transcriptions"
curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
  -F "file=@$AUDIO" -F response_format=verbose_json \
  | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print('text:', d.get('text','')[:400]); print('words:', len(d.get('words',[])), 'segments:', len(d.get('segments',[])))"

echo
echo "== 4/4 streaming WebSocket /asr (realtime playback)"
.venv/bin/python scripts/stream_test.py "$AUDIO" --url "ws://127.0.0.1:$PORT/asr" --realtime 1.0

echo
echo "verify: OK"
