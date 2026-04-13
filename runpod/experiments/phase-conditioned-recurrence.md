# Phase-Conditioned Recurrence

Branch: `phase-conditioned-recurrence`

## Thesis

Shared-block recurrence often underuses depth because each pass behaves too similarly. This branch adds explicit phase conditioning so the same block can operate in different modes across passes.

## Change

- Adds `PHASE_COUNT` and `PHASE_MOD_STRENGTH`.
- Learns a small phase embedding table.
- Each recurrent step maps to a phase ID (`step % PHASE_COUNT`).
- The phase vector modulates attention and MLP residual scales in opposite directions:
  - attention scale multiplied by `(1 + phase_mod)`
  - MLP scale multiplied by `(1 - phase_mod)`

## Why It Might Work

- Encourages role separation across recurrent passes without untieing full layers.
- Keeps parameter growth tiny while allowing the shared block to specialize by phase.

## Suggested First Run

Default is `4` GPUs because [config-1.sh](C:/Users/yuvar/OneDrive/Documents/GitHub/parameter-golf-thinker/runpod/phase-conditioned-recurrence/config-1.sh) sets `NPROC_PER_NODE=4` unless overridden.

One-paste run (train + save logs/artifacts):

```bash
NPROC_PER_NODE=4 \
PHASE_COUNT=4 \
PHASE_MOD_STRENGTH=0.2 \
ITERATIONS=5000 \
VAL_LOSS_EVERY=500 \
bash runpod/phase-conditioned-recurrence/config-1.sh
```

This produces:
- `logs/<RUN_ID>.txt`
- `logs/<RUN_ID>.console.log`
- `logs/<RUN_ID>_adaptive_eval.jsonl`
- `logs/<RUN_ID>_adaptive_eval.csv`
- `runpod/experiments/<date>-<gpu>-<count>gpu/<RUN_ID>/...` via `./save.sh`
