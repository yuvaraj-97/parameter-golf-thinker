# Adaptive Cascade Controller

This document defines the first adaptive training controller for `parameter-golf-thinker`.

## Goal

Use a fast default regime early, then switch to more quality-oriented regimes only after the validation curve flattens enough to justify the extra overhead.

Initial cascade:

1. `master`
2. `late_ema`
3. `late_qat`

This repository change adds:

- adaptive eval cadence
- flattening detection
- late EMA activation
- late QAT activation
- JSONL checkpoint history for reuse in later runs

## Eval cadence

- Default coarse cadence: every `VAL_LOSS_EVERY` steps
- Optional dense cadence: every `ADAPTIVE_EVAL_DENSE_EVERY` steps
- Dense cadence only applies inside a local band around a suspected flattening point

### Dense band source

The dense band is centered on:

1. `ADAPTIVE_EVAL_BAND_CENTER_STEP` if explicitly provided
2. otherwise, the latest `flatten_step` recovered from `ADAPTIVE_EVAL_HISTORY_PATH`

Dense band width:

- `ADAPTIVE_EVAL_BAND_RADIUS_STEPS`

## Flattening detector

At each validation checkpoint:

1. store `step`, `elapsed_ms`, `val_loss`, `val_bpb`, `step_avg_ms`
2. compute per-interval `val_bpb` improvement per step
3. compare the average of the most recent `ADAPTIVE_EVAL_FLATTEN_WINDOW` intervals against the best rate seen so far

Flattening is detected when:

- recent mean gain rate <= `best_gain_rate * ADAPTIVE_EVAL_FLATTEN_RATIO`

This keeps the trigger relative rather than hard-coding one absolute threshold.

## Stored history

The controller writes JSONL records to:

- `logs/<run_id>_adaptive_eval.jsonl`

Each eval record includes:

- `run_id`
- `step`
- `train_time_ms`
- `val_loss`
- `val_bpb`
- `step_avg_ms`
- `dense_band_center_step`
- `flatten_step` when detected

This history is intended to inform the next run's dense band placement.

## Current stage behavior

1. `base`
   - normal training
2. `late_ema`
   - EMA shadow weights update every step
   - eval/export use EMA weights
3. `late_qat`
   - EMA remains active
   - `CastedLinear` layers use row-wise fake quantization in forward

The remaining open problem is threshold tuning, not the stage wiring itself.
