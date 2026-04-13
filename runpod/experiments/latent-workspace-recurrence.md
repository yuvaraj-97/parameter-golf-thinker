# Latent Workspace Recurrence

Branch: `codex/latent-workspace-recurrence`

## Thesis

Plain shared-block recurrence keeps reapplying the same transformation to token states. This branch adds a tiny persistent latent workspace so recurrent passes can accumulate scratch state outside the token stream.

## Change

- Adds `LATENT_SLOTS` learned workspace vectors.
- Each recurrent pass:
  - pools the current token state into a global summary
  - writes that summary into the latent workspace
  - updates the workspace with a tiny self-transition
  - writes a distilled workspace signal back into token states before the shared block

## Why It Might Work

- Gives the recurrent model an explicit scratchpad rather than forcing all iterative reasoning into ordinary token residuals.
- Keeps parameter cost small relative to adding more untied transformer layers.
- Preserves the repo's central bet: more effective computation per stored parameter.

## Suggested First Run

Default is `4` GPUs because [config-1.sh](C:/Users/yuvar/OneDrive/Documents/GitHub/parameter-golf-thinker/runpod/latent-workspace-recurrence/config-1.sh) sets `NPROC_PER_NODE=4` unless you override it.

One-paste run (train + save logs/artifacts):

```bash
NPROC_PER_NODE=4 \
LATENT_SLOTS=8 \
ITERATIONS=5000 \
VAL_LOSS_EVERY=500 \
bash runpod/latent-workspace-recurrence/config-1.sh
```

This produces:
- `logs/<RUN_ID>.txt`
- `logs/<RUN_ID>.console.log`
- `logs/<RUN_ID>_adaptive_eval.jsonl`
- `logs/<RUN_ID>_adaptive_eval.csv`
- `runpod/experiments/<date>-<gpu>-<count>gpu/<RUN_ID>/...` via `./save.sh`
