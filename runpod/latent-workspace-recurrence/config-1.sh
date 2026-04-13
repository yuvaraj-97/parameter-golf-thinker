#!/bin/sh
set -eu

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export LATENT_SLOTS="${LATENT_SLOTS:-8}"
export ITERATIONS="${ITERATIONS:-10000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-400}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

if [ $((ITERATIONS % 1000)) -eq 0 ]; then
  ITER_TAG="$((ITERATIONS / 1000))k"
else
  ITER_TAG="${ITERATIONS}"
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
export RUN_ID="${RUN_ID:-latent_workspace_${NPROC_PER_NODE}gpu_${ITER_TAG}_${STAMP}}"

mkdir -p logs

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py 2>&1 | tee "logs/${RUN_ID}.console.log"
bash ./save.sh
