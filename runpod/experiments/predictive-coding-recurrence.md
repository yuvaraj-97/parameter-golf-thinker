# Predictive-Coding Recurrence

Branch: `codex/predictive-coding-recurrence`

## Thesis

Standard recurrence replaces token states each pass. This branch treats each pass as an error-correction step that predicts only the residual update.

## Change

- Adds `PREDICTIVE_DELTA_STRENGTH` and `PREDICTIVE_DELTA_CLAMP`.
- Keeps the shared recurrent block, but computes:
  - `candidate = shared_block(x, x0)`
  - `delta = candidate - x`
  - learned projection + tanh-clipped correction
- Applies the correction with learned per-dimension gain instead of full state replacement.

## Why It Might Work

- Matches iterative inference behavior where each pass should refine errors rather than rewrite the full representation.
- Improves recurrence stability by constraining per-step updates.

## Suggested First Run

```bash
RUN_ID=predictive_coding_smoke \
PREDICTIVE_DELTA_STRENGTH=0.5 \
PREDICTIVE_DELTA_CLAMP=2.0 \
ITERATIONS=1000 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
