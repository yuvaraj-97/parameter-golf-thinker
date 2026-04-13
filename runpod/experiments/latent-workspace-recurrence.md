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

```bash
RUN_ID=latent_workspace_smoke \
LATENT_SLOTS=8 \
ITERATIONS=1000 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
