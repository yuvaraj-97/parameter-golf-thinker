# Phase-Conditioned Recurrence

Branch: `codex/phase-conditioned-recurrence`

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

```bash
RUN_ID=phase_conditioned_smoke \
PHASE_COUNT=4 \
PHASE_MOD_STRENGTH=0.2 \
ITERATIONS=1000 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
