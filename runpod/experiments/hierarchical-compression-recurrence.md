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

```bash
RUN_ID=hierarchical_compression_smoke \
COMPRESSION_INTERVAL=3 \
COMPRESSION_GROUPS=8 \
ITERATIONS=1000 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
