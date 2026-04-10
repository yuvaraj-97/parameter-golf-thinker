#!/bin/sh
set -eu

RUN_ID="${RUN_ID:-main_baseline}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_gpt.py
