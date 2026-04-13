# Hierarchical Compression Recurrence

Branch: `codex/hierarchical-compression-recurrence`

## Thesis

Recurrent passes at full token resolution are expensive and redundant. This branch periodically compresses token states into coarse groups, refines those summaries, then projects the signal back.

## Change

- Adds `COMPRESSION_INTERVAL` and `COMPRESSION_GROUPS`.
- After every `COMPRESSION_INTERVAL` recurrent passes:
  - mean-pools tokens into `COMPRESSION_GROUPS` groups
  - applies learned compress/decompress projections
  - broadcasts the compressed representation back to token resolution
- Uses additive residual injection so the original token stream is preserved.

## Why It Might Work

- Introduces multi-scale refinement inside recurrence without adding full untied layers.
- Encourages global information mixing at lower effective resolution.

## Suggested First Run

Default is `4` GPUs because [config-1.sh](C:/Users/yuvar/OneDrive/Documents/GitHub/parameter-golf-thinker/runpod/hierarchical-compression-recurrence/config-1.sh) sets `NPROC_PER_NODE=4` unless overridden.

One-paste run (train + save logs/artifacts):

```bash
NPROC_PER_NODE=4 \
COMPRESSION_INTERVAL=3 \
COMPRESSION_GROUPS=8 \
ITERATIONS=5000 \
VAL_LOSS_EVERY=500 \
bash runpod/hierarchical-compression-recurrence/config-1.sh
```

This produces:
- `logs/<RUN_ID>.txt`
- `logs/<RUN_ID>.console.log`
- `logs/<RUN_ID>_adaptive_eval.jsonl`
- `logs/<RUN_ID>_adaptive_eval.csv`
- `runpod/experiments/<date>-<gpu>-<count>gpu/<RUN_ID>/...` via `./save.sh`
