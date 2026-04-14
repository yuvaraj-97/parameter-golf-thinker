# Parameter-Golf Recovery Plan (2026-04-14)

Canonical execution plan for the next optimization cycle.

## Summary

Run a strict funnel:

1. Reproducibility
2. Single-idea validation
3. Compound experiments
4. 8x H100 scale-up

Tracking model:

- GitHub Issues + Milestones
- Explicit dependency links in issue body: `blocked by #<issue_number>`
- One parent meta-issue per milestone

## Baseline Locks

- `master_8gpu_5000` roundtrip `val_bpb=1.4932`
- `master_8gpu_20000` roundtrip `val_bpb=1.4785`
- Reference: [summary.json](./2026-04-10-runpod-baselines/summary.json)

## Run Matrix

| Stage | Branch Set | Commit SHA | GPU Type/Count | Iterations | Eval Cadence | Expected Wallclock | Pass/Fail Gate |
|---|---|---|---|---:|---|---|---|
| Smoke | Current branch under test | pin in run README | 1x A4000 or 1x 4090 | 200 | `VAL_LOSS_EVERY=50` | ~minutes | No crash, no NaN, logs + artifacts saved |
| Screening | `latent`, `predictive`, `phase`, `hierarchical` (post-repro) | pin in run README | 4x 4090 (preferred) | 5000 | `VAL_LOSS_EVERY=500` | ~20-35 min per run (hardware dependent) | Roundtrip `< 1.4932` and stable/improving trend |
| Promotion | Top 1-2 from screening | pin in run README | 4x 4090 | 10000 | `VAL_LOSS_EVERY=1000` | ~40-70 min per run | Better than own 5k and better than `master_5k` lock |
| Pre-final | Winner (single or compound) | pin in run README | 8x H100 | 10000 | `VAL_LOSS_EVERY=1000` + final eval | ~10-20 min (estimate) | Reproduces 4x trend on target hardware |
| Final target | Winner | pin in run README | 8x H100 | 20000 (+optional tail) | sparse periodic (target `2000`) + final eval | ~20-40 min (estimate) | Material improvement vs `1.4785`, target toward `<=1.2244` |

## Strategy Decisions

Implement now:

- Lower-LR tail schedule
- Strict ablation protocol

Implement later:

- Sequence packing/length bucketing (only if variable-length batching is introduced)
- Optimizer replacement (after architecture + schedule winners are clear)

Compound policy:

- Allow only evidence-backed compounds
- Start with `latent + predictive`
- Block hierarchical compounding until reproducibility is fixed

## Milestones, Issues, and Dependency Order

## GitHub Tracking (Created)

Milestones:

- `M0-Reproducibility` (ID 1)
- `M1-Single-Idea Validation` (ID 2)
- `M2-Compound Experiments` (ID 3)
- `M3-H100 Scale-Up` (ID 4)

Issues:

- `#1` M0 parent: [M0-Reproducibility: parent tracker](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/1)
- `#2` A: [Standardize run protocol (env/tokenizer/save/log/seed)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/2)
- `#3` B: [Re-run hierarchical-compression twice on identical hardware/config](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/3)
- `#4` C: [Baseline lock check (master 5k + 20k parity vs summary.json)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/4)
- `#5` M1 parent: [M1-Single-Idea Validation: parent tracker](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/5)
- `#6` D: [latent-workspace-recurrence 5k screening run (4x4090)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/6)
- `#7` E: [predictive-coding-recurrence 5k screening run (4x4090)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/7)
- `#8` F: [phase-conditioned-recurrence 5k screening run (control/comparison)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/8)
- `#9` M2 parent: [M2-Compound Experiments: parent tracker](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/9)
- `#10` G: [Create compound-latent-predictive branch from winning base](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/10)
- `#11` H: [10k ablation set (winner base vs +latent vs +predictive vs compound)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/11)
- `#12` M3 parent: [M3-H100 Scale-Up: parent tracker](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/12)
- `#13` I: [8x H100 10k confirmation run (winner branch)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/13)
- `#14` J: [8x H100 20k target run (winner branch)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/14)
- `#15` K: [Optional 8x H100 extended-tail run (conditional on 20k slope)](https://github.com/yuvaraj-97/parameter-golf-thinker/issues/15)

### M0-Reproducibility

- Meta issue: `M0-Reproducibility: parent tracker`
- Issue A: Standardize run protocol (fixed env, tokenizer path, save path, log naming, seed policy)
- Issue B: Re-run `hierarchical-compression-recurrence` twice on identical hardware/config to resolve conflicting logs
- Issue C: Baseline lock check (`master` 5k + 20k parity vs `summary.json`)

Completion gate:

- Reproducibility confirmed and baseline parity verified

### M1-Single-Idea Validation (blocked by M0)

- Meta issue: `M1-Single-Idea Validation: parent tracker`
- Issue D: `latent-workspace-recurrence` screening run
- Issue E: `predictive-coding-recurrence` screening run
- Issue F: `phase-conditioned-recurrence` screening run (control/comparison)

Completion gate:

- At least one branch credibly beats `master_5k` lock

### M2-Compound Experiments (blocked by M1)

- Meta issue: `M2-Compound Experiments: parent tracker`
- Issue G: Create `compound-latent-predictive` branch from winner base
- Issue H: Ablation set (base winner vs +latent vs +predictive vs compound)

Completion gate:

- Compound beats best single-idea branch at 10k

### M3-H100 Scale-Up (blocked by M2)

- Meta issue: `M3-H100 Scale-Up: parent tracker`
- Issue I: 8x H100 10k confirmation run
- Issue J: 8x H100 20k target run
- Issue K: Optional 8x H100 extended-tail run (only if 20k slope still improving)

Completion gate:

- 20k H100 run completes with stable curve and improved roundtrip

## Required Artifacts Per Run

Each run issue is not complete unless all files exist:

- `train.log`
- `adaptive_eval.csv`
- `adaptive_eval.jsonl`
- `final_model.int8.ptz`
- experiment `README.md`

Each promoted run must also include:

- Explicit commit SHA
- Same tokenizer/data path as phase peers
- Same eval policy within phase

## Execution Notes

- Keep 4x screening on one GPU class (`4090`) to reduce confounding.
- Do not compare 4080 SUPER and 4090 as if equivalent for winner selection.
- On H100, sparse eval is allowed during training, but final roundtrip eval is mandatory.
