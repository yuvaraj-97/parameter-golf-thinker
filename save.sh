#!/usr/bin/env bash
set -euo pipefail

# Auto-detect RUN_ID from the most recently modified adaptive_eval.jsonl if not set.
if [ -z "${RUN_ID:-}" ]; then
  latest="$(ls -t logs/*_adaptive_eval.jsonl 2>/dev/null | head -n 1)"
  if [ -z "$latest" ]; then
    echo "error: RUN_ID is not set and no adaptive_eval.jsonl found in logs/" >&2
    exit 1
  fi
  RUN_ID="$(basename "$latest" _adaptive_eval.jsonl)"
  echo "auto-detected RUN_ID: $RUN_ID"
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
GPU_SLUG="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\+/-/g; s/^-//; s/-$//')"
DATE_TAG="$(date +%F)"
POD_NAME="$(hostname)"
RUN_DIR="runpod/experiments/${DATE_TAG}-${GPU_SLUG}-${GPU_COUNT}gpu/${RUN_ID}"

mkdir -p "$RUN_DIR"

# Required log files.
cp "logs/${RUN_ID}.txt" "$RUN_DIR/train.log"
cp "logs/${RUN_ID}_adaptive_eval.jsonl" "$RUN_DIR/adaptive_eval.jsonl"
cp "logs/${RUN_ID}_adaptive_eval.csv" "$RUN_DIR/adaptive_eval.csv"

# Optional files.
[ -f "logs/${RUN_ID}.console.log" ] && cp "logs/${RUN_ID}.console.log" "$RUN_DIR/console.log" || echo "skipping missing: console.log"
[ -f final_model.pt ] && cp final_model.pt "$RUN_DIR/final_model.pt" || echo "skipping missing: final_model.pt"
[ -f final_model.int8.ptz ] && cp final_model.int8.ptz "$RUN_DIR/final_model.int8.ptz" || echo "skipping missing: final_model.int8.ptz"

printf '# %s\n\n- Pod: %s\n- GPU: %sx %s\n- Run ID: `%s`\n' \
  "$RUN_ID" "$POD_NAME" "$GPU_COUNT" "$GPU_NAME" "$RUN_ID" > "$RUN_DIR/README.md"

git restore final_model.pt final_model.int8.ptz 2>/dev/null || true
git add "$RUN_DIR"
git commit -m "logs: add ${RUN_ID} ${GPU_SLUG} ${GPU_COUNT}gpu run"
git push origin "$(git branch --show-current)"
