#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?RUN_ID is not set}"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
GPU_SLUG="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\+/-/g; s/^-//; s/-$//')"
DATE_TAG="$(date +%F)"
POD_NAME="$(hostname)"
RUN_DIR="runpod/experiments/${DATE_TAG}-${GPU_SLUG}-${GPU_COUNT}gpu/${RUN_ID}"

mkdir -p "$RUN_DIR"

cp "logs/${RUN_ID}.console.log" "$RUN_DIR/console.log"
cp "logs/${RUN_ID}.txt" "$RUN_DIR/train.log"
cp "logs/${RUN_ID}_adaptive_eval.jsonl" "$RUN_DIR/adaptive_eval.jsonl"
cp "logs/${RUN_ID}_adaptive_eval.csv" "$RUN_DIR/adaptive_eval.csv"

if [ -f final_model.pt ]; then
  cp final_model.pt "$RUN_DIR/final_model.pt"
fi

if [ -f final_model.int8.ptz ]; then
  cp final_model.int8.ptz "$RUN_DIR/final_model.int8.ptz"
fi

printf '# %s\n\n- Pod: %s\n- GPU: %sx %s\n- Run ID: `%s`\n' \
  "$RUN_ID" "$POD_NAME" "$GPU_COUNT" "$GPU_NAME" "$RUN_ID" > "$RUN_DIR/README.md"

git restore final_model.pt final_model.int8.ptz 2>/dev/null || true
git add "$RUN_DIR"
git commit -m "logs: add ${RUN_ID} ${GPU_SLUG} ${GPU_COUNT}gpu run"
git push origin adaptive-cascade-controller
