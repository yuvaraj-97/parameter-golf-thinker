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

Default is `4` GPUs because [config-1.sh](C:/Users/yuvar/OneDrive/Documents/GitHub/parameter-golf-thinker/runpod/predictive-coding-recurrence/config-1.sh) sets `NPROC_PER_NODE=4` unless overridden.

One-paste run (train + save logs/artifacts):

```bash
NPROC_PER_NODE=4 \
PREDICTIVE_DELTA_STRENGTH=0.5 \
PREDICTIVE_DELTA_CLAMP=2.0 \
ITERATIONS=5000 \
VAL_LOSS_EVERY=500 \
bash runpod/predictive-coding-recurrence/config-1.sh
```

This produces:
- `logs/<RUN_ID>.txt`
- `logs/<RUN_ID>.console.log`
- `logs/<RUN_ID>_adaptive_eval.jsonl`
- `logs/<RUN_ID>_adaptive_eval.csv`
- `runpod/experiments/<date>-<gpu>-<count>gpu/<RUN_ID>/...` via `./save.sh`
