# Session Working Note - 2026-04-10

This document captures the working context from the Runpod and adaptive-cascade session so a future chat can continue without relying on thread history. It was written before an actual handoff happened, so treat it as a live working note rather than a closed transfer memo.

## Runpod Workflow Status

### SSH and pod workflow

- CPU pod SSH flow was verified successfully with Termius.
- Official image used for real runs: `runpod/parameter-golf:latest`
- Reusing existing SSH key `id_ed25519` worked.
- Direct TCP SSH and Runpod relay SSH were both discussed; direct TCP with `root` was the normal path for GPU pods.

### Templates created

- `parameter-golf-train-1gpu`
  - Runpod template ID: `8drpuqso4j`
  - README includes clone and smoke commands for `master` and `feature/memory-optimizations`

Other earlier templates were created during the session, but the 1-GPU template above is the one explicitly preserved for ongoing work.

### 2026-04-11 template hardening

The `parameter-golf-train-1gpu` template was hardened so a fresh pod now boots much closer to a ready-to-train state.

Environment variables added:

- `GITHUB_SSH_PRIVATE_KEY_B64`
- `GIT_USER_NAME=yuvaraj-97`
- `GIT_USER_EMAIL=mailtoyuvaraj11@gmail.com`
- `GIT_BRANCH=adaptive-cascade-controller`

Important detail:

- `GITHUB_SSH_PRIVATE_KEY_B64` is bound to a Runpod secret containing the base64-encoded SSH private key, not the raw multiline key.

Container bootstrap behavior now:

- decodes `GITHUB_SSH_PRIVATE_KEY_B64` into `/root/.ssh/id_ed25519`
- fixes SSH permissions
- derives `/root/.ssh/id_ed25519.pub`
- adds GitHub to `known_hosts`
- sets global git username and email
- removes the image-default `/workspace/parameter-golf`
- clones `git@github.com:yuvaraj-97/parameter-golf-thinker.git`
- checks out and fast-forwards `adaptive-cascade-controller`
- attempts dataset preparation
- leaves the container alive via `sleep infinity`

Result:

- a fresh pod should already have GitHub SSH configured
- the repository cloned
- the target branch checked out
- dataset preparation attempted
- the shell ready to start training immediately

## Repo branches tested on Runpod

### `master`

Validated successfully on:

- `1x RTX 2000 Ada`
- `8x H100 80GB HBM3`

### `feature/memory-optimizations`

- Ran successfully on `1x RTX 2000 Ada`
- Used much less VRAM than `master`
- Was slower and worse than `master` in the tested smoke run
- Kept intentionally for later / end-stage consideration rather than as current default

## Confirmed results

### 1 GPU, 200 steps, `master`

- step-200 `val_bpb`: `2.3237`
- final roundtrip `val_bpb`: `2.3739`
- train time: `45393 ms`
- eval time: `145865 ms`
- peak memory allocated: `697 MiB`

### 1 GPU, 200 steps, `feature/memory-optimizations`

- step-200 `val_bpb`: `2.3627`
- final roundtrip `val_bpb`: `2.3999`
- train time: `104726 ms`
- eval time: `354486 ms`
- peak memory allocated: `242 MiB`

### 8 GPU, 200 steps, `master`

- step-200 `val_bpb`: `2.1647`
- final roundtrip `val_bpb`: `2.2756`
- train time: `5887 ms`
- eval time: `2739 ms`

### 8 GPU, 5000 steps, `master`

- step-5000 `val_bpb`: `1.4847`
- final roundtrip `val_bpb`: `1.4932`
- train time: `147561 ms`
- eval time: `2655 ms`

### 8 GPU, 20000 steps, `master`

- step-20000 `val_bpb`: `1.4589`
- final roundtrip `val_bpb`: `1.4785`
- train time: `612766 ms` (`10m 12.8s`)
- eval time: `2670 ms`
- note: slightly above a strict 10-minute training budget

## Telemetry conclusions

### 1x RTX 2000 Ada

- Active training used relatively little VRAM.
- The tested setups looked compute-bound rather than memory-bound.

### 1x RTX A4000 Community Cloud

Telemetry snapshot captured during active single-GPU adaptive smoke testing on 2026-04-11:

- Pod: `broken_indigo_dragonfly`
- Pod ID: `r8dhfu3ey333uv`
- Uptime at capture: `10m 26s`
- Disk: `446 MB / 50 GB`
- CPU load: `8%`
- Host CPU: `AMD EPYC 7453`
- Host RAM: `3.943 GiB / 57.74 GiB`
- GPU: `RTX A4000`
- GPU utilization: `62%`
- GPU VRAM: `1 GiB / 16 GiB`
- GPU temperature: `86 C`
- GPU power draw: `138 W`
- Driver version: `550.144.03`
- CUDA version: `12.4`

Interpretation:

- the smoke run fits very comfortably in `16 GB` VRAM
- host RAM and CPU are not bottlenecks
- this setup still looks compute-bound rather than memory-bound

### 8x H100 80GB HBM3

- GPU utilization reached 100%.
- VRAM usage was only around `~7 GiB / 80 GiB`.
- CPU, host RAM, and disk were not bottlenecks.
- Conclusion: current `master` configuration on H100 is compute-bound, not memory-bound.

Implication:

- pure memory-saving work is not currently justified on H100 unless it unlocks larger batch, longer context, or a larger model

## Experiment logging added to repo

Structured experiment notes were added under:

- [runpod/experiments/2026-04-10-runpod-baselines/README.md](C:\Users\yuvar\OneDrive\Documents\GitHub\parameter-golf-thinker\runpod\experiments\2026-04-10-runpod-baselines\README.md)
- [runpod/experiments/2026-04-10-runpod-baselines/summary.json](C:\Users\yuvar\OneDrive\Documents\GitHub\parameter-golf-thinker\runpod\experiments\2026-04-10-runpod-baselines\summary.json)

These contain:

- run summaries
- metrics
- telemetry observations
- next experiment hypotheses

## Adaptive cascade work

### Intent

The new idea was to avoid one fixed training regime for the full run. Instead:

1. start with fast base training
2. monitor periodic validation
3. detect flattening
4. switch into more quality-oriented modes late
5. reuse previous run history to place future dense-eval bands earlier

### Agreed cascade

- `base`
- `late_ema`
- `late_qat`

### Eval strategy agreed

- coarse eval every `500` iterations by default
- dense eval every `100` iterations only near a suspected flattening band
- use previous runs to move the dense band earlier in future runs

### Files changed for adaptive cascade

- [train_gpt.py](C:\Users\yuvar\OneDrive\Documents\GitHub\parameter-golf-thinker\train_gpt.py)
- [runpod/experiments/adaptive-cascade-controller.md](C:\Users\yuvar\OneDrive\Documents\GitHub\parameter-golf-thinker\runpod\experiments\adaptive-cascade-controller.md)

### What is already implemented in `train_gpt.py`

- adaptive eval cadence
- optional dense eval band
- flattening detector
- JSONL eval history logging
- prior-history reuse to infer the next dense-band center
- stage tracking:
  - `base`
  - `late_ema`
  - `late_qat`
- EMA shadow updates after the stage switch
- EMA-based eval/export once EMA stage is active
- simple row-wise fake-quant forward path for late QAT in `CastedLinear`

### What is not yet validated

- No serious end-to-end adaptive-cascade run has been validated yet.
- The code compiles, but the controller and stage switches still need smoke testing.
- This branch should be validated on a cheap 1-GPU pod first before any multi-GPU adaptive run.

### 2026-04-11 smoke findings

Observed during the first A4000 adaptive smoke:

- dense adaptive evals fired at `step 50`, `75`, and `100`
- `flatten_detected` fired at `step 100`
- the controller switched from `base` to `late_ema` at `step 100`
- the first `late_ema` eval regressed sharply from `2.5528` to `3.8331 val_bpb`

Interpretation:

- the smoke configuration was aggressive enough to trigger flattening too early for a `200`-step run
- more importantly, the first `late_ema` eval exposed a controller bug: EMA state was initialized at startup and not re-seeded from live weights when switching stages

Repo fix applied after this finding:

- re-seed `ema_shadow` from `base_model` at the moment of the `base -> late_ema` transition
- track flatten detection per stage instead of resetting `flatten_step` after every eval record

This fix should be present before trusting any subsequent `late_ema` or `late_qat` smoke conclusions.

## Branching / git status note

- The user created a new branch named: `adaptive-cascade-controller`
- The assistant could not create local branches directly because `.git` ref writes were permission-blocked in this workspace
- The expectation was: user creates the branch manually and then applies/commits the diff there

## Recommended next session start

1. Check current branch and diff:
   - `git branch --show-current`
   - `git status`
   - `git diff`
2. Review the adaptive-cascade files listed above
3. Run a cheap 1-GPU smoke test for the cascade controller first
4. Verify:
   - adaptive eval logs appear
   - flatten detection fires
   - `late_ema` switch happens
   - optional `late_qat` switch happens
   - final eval completes
5. Only after that move the adaptive controller to 8-GPU testing

## Suggested first adaptive smoke command

This was the recommended correctness-first smoke run for the new branch:

```bash
RUN_ID=adaptive_cascade_smoke_1gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=32768 \
VAL_BATCH_SIZE=8192 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=50 \
ADAPTIVE_EVAL_ENABLED=1 \
ADAPTIVE_EVAL_DENSE_EVERY=25 \
ADAPTIVE_EVAL_BAND_CENTER_STEP=100 \
ADAPTIVE_EVAL_BAND_RADIUS_STEPS=50 \
ADAPTIVE_EVAL_FLATTEN_WINDOW=2 \
ADAPTIVE_EVAL_FLATTEN_RATIO=0.5 \
ADAPTIVE_EVAL_MIN_REMAINING_FRACTION=0.3 \
CASCADE_ENABLED=1 \
QAT_ENABLED=1 \
EMA_DECAY=0.997 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee adaptive-cascade-smoke-1gpu.log
```

## Current strategic direction

- `master` is still the default training branch
- `feature/memory-optimizations` is deferred for late-stage or specialized exploration
- H100 telemetry suggests the next important work is quality-per-step and adaptive schedules, not raw memory reduction
