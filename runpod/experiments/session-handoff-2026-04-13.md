# Session Working Note - 2026-04-13

This document continues from `session-handoff-2026-04-10.md`. It captures the working context from the adaptive-cascade debugging and QAT LR fix session so a future chat can continue without relying on thread history.

## Critical bugs found and fixed this session

Three bugs in the cascade controller were identified through live experiment analysis.

### 1. QAT running at full learning rate (most impactful)

**Commit:** `67eda94` fix: QAT LR reduction + early stop on QAT plateau

**Problem:** The cascade controller triggered QAT on EMA flatten detection regardless of LR state. In the 5k QAT run (`adaptive_cascade_4gpu_5k_qat_20260413-042201`), QAT activated at step 2400 with LR scale=1.0. The STE fake-quantize noise at full LR destabilized training: val_bpb degraded from 1.5137 to 1.6476 over 2600 steps.

**Root cause:** The old record-setting runs (2026-03-22, val_bpb=1.1228) used `LATE_QAT_THRESHOLD=0.15` which only activated QAT when the LR had already dropped below 15%. The cascade controller had no equivalent mechanism.

**Fix:** Added `QAT_LR_FRACTION` env var (default 0.15, calibrated from old records). When `_QAT_ACTIVE` is True, the effective LR = `lr_mul(step) * qat_lr_fraction`. This matches the old threshold behavior: QAT runs at ~15% of peak LR.

**Evidence from old records:**

| Run | QAT activation scale | Final int8 val_bpb |
|-----|---------------------|-------------------|
| 2026-03-22 (warmdown=3500, iter=20k) | 0.15 | **1.1228** (record) |
| 2026-03-21 (warmdown=1200, iter=9k) | 0.10 | 1.1248 |

### 2. Missing QAT early stop handler

**Commit:** `67eda94` (same commit)

**Problem:** When `late_qat` stage flattened, there was no handler for `stage_index == 2`. The code only handled stage 0 (base) and stage 1 (late_ema). QAT flatten detection fired (`adaptive_eval:flatten_detected step:2900 stage:late_qat`) but no action was taken.

**Fix:** Added `elif args.cascade_enabled and stage_index == 2:` clause that sets `stop_after_step = step` when QAT plateaus.

### 3. Flatten detector blind to stages that never improve

**Commit:** `ac87d95` fix: detect_flattening returns True when stage never improved

**Problem:** In the 10k run (`adaptive_cascade_4gpu_10k_qat_20260413-051449`), the late_ema stage peaked at step 2500 (val_bpb=1.5128) before the first stage eval was recorded. Every gain_rate in the stage history was negative. `detect_flattening` had `if best_rate <= 0: return False` — it assumed there must be initial improvement before flattening. The detector never fired, and late_ema degraded from 1.5128 to 1.6027+ over 2200+ wasted steps.

**Fix:** Changed `return False` to `return True`. If after `flatten_window + 2` data points the model has never improved within a stage, it's past-plateau. The minimum history requirement (5 entries) prevents false positives from noise.

## Config changes

**Commit:** `147fec5` feat: auto-detect dense eval band from history

`config-1.sh` updated:

- `ITERATIONS`: 5000 → **10000**
- `QAT_ENABLED`: 0 → **1**
- `QAT_LR_FRACTION`: **0.15** (new)
- `VAL_LOSS_EVERY`: 500 → **1000** (coarse eval, scaled for longer runs)
- `ADAPTIVE_EVAL_DENSE_EVERY`: 100 → **200**
- `TRAIN_LOG_EVERY`: 200 → **400**
- `ADAPTIVE_EVAL_BAND_CENTER_STEP`: 5000 → **0** (auto-detect from history)
- `ADAPTIVE_EVAL_BAND_RADIUS_STEPS`: 1500 → **2000**
- `ADAPTIVE_EVAL_HISTORY_PATH`: new, empty by default

Band auto-detection cascade:
1. If `ADAPTIVE_EVAL_BAND_CENTER_STEP > 0`: use it directly
2. Else if `ADAPTIVE_EVAL_HISTORY_PATH` points to a previous run's jsonl: read the latest `flatten_step` from it
3. Else: fall back to `iterations / 5` (catches base flatten at ~20% of training, consistent with observed step 1800-2000 across runs)

## Experiment results this session

### 4x RTX 4090, 5k steps, QAT enabled (before LR fix)

- Run ID: `adaptive_cascade_4gpu_5k_qat_20260413-042201`
- Base flatten: step 1800
- EMA flatten / QAT switch: step 2400
- QAT best val_bpb: **1.5137** (step 2600)
- QAT final val_bpb: 1.6476 (step 5000, degraded due to full LR)
- Final int8 roundtrip: **1.6527**
- Diagnosis: QAT at full LR, missing early stop

### 4x RTX 4090, 10k steps, QAT enabled (before flatten fix)

- Run ID: `adaptive_cascade_4gpu_10k_qat_20260413-051449`
- Base flatten: step 2000
- EMA best val_bpb: **1.5128** (step 2500)
- EMA at step 4700: 1.6027 (still degrading, detector never fired)
- Run saved/stopped — needs restart with fixed code
- Diagnosis: `detect_flattening` returned False for all-negative gain rates

### Previous session results (carried forward)

#### 4x RTX 5000 Ada, 5k steps, no QAT

- Run ID: `adaptive_cascade_4gpu_5k_no_qat_20260411-182722`
- Base flatten: step 1800
- EMA best val_bpb: **1.5170** (step 2300)
- EMA at step 3000: 1.5309 (degrading past optimum, no early stop in old code)
- Did not complete to serialization

#### 4x RTX 5000 Ada, 5k steps, QAT enabled (early version)

- Run ID: `adaptive_cascade_4gpu_5k_20260411-171435`
- Final int8 roundtrip: **1.6810**

## Runpod infrastructure (unchanged from 2026-04-10)

### SSH and pod workflow

- CPU pod SSH flow verified with Termius
- Official image: `runpod/parameter-golf:latest`
- Direct TCP SSH with `root` is the normal path for GPU pods

### Template

- `parameter-golf-train-1gpu` (Runpod template ID: `8drpuqso4j`)
- Hardened with auto-bootstrap: SSH key decode, git clone, branch checkout, dataset prep

### Template env vars

- `GITHUB_SSH_PRIVATE_KEY_B64` (bound to Runpod secret, base64-encoded)
- `GIT_USER_NAME=yuvaraj-97`
- `GIT_USER_EMAIL=mailtoyuvaraj11@gmail.com`
- `GIT_BRANCH=adaptive-cascade-controller`

## Confirmed baseline results (from 2026-04-10)

### 8 GPU H100, master branch

| Steps | val_bpb | int8 roundtrip | Train time |
|-------|---------|---------------|------------|
| 200 | 2.1647 | 2.2756 | 5.9s |
| 5000 | 1.4847 | 1.4932 | 2m 27.6s |
| 20000 | 1.4589 | 1.4785 | 10m 12.8s |

### Record-setting runs (from record archives)

| Date | Config | int8 val_bpb | Key detail |
|------|--------|-------------|------------|
| 2026-03-22 | 11L EMA GPTQ-lite warmdown3500 QAT015 | **1.1228** | QAT at scale 0.15, 20k iter wallclock-capped |
| 2026-03-21 | 11L XSA4 EMA PartialRoPE LateQAT | **1.1248** | QAT at scale 0.10, 9k iter wallclock-capped |

## Current branch status

- Active branch: `adaptive-cascade-controller`
- All three bugs fixed and committed
- Config-1 updated for 10k with QAT + history-based band detection
- Needs a fresh 10k run with the fixed code to validate

## Recommended next session start

1. Push the branch to remote if not already done:
   ```bash
   git push origin adaptive-cascade-controller
   ```

2. Start a fresh 10k run on RunPod with fixed code:
   ```bash
   export RUN_ID="adaptive_cascade_4gpu_10k_qat_$(date +%Y%m%d-%H%M%S)"
   bash runpod/adaptive-cascade-controller/config-1.sh
   ```

3. Optionally pass history from previous run:
   ```bash
   export ADAPTIVE_EVAL_HISTORY_PATH="logs/adaptive_cascade_4gpu_5k_qat_20260413-042201_adaptive_eval.jsonl"
   ```

4. Verify in logs:
   - `cascade:enabled ... qat_lr_fraction:0.150` appears at start
   - Base flatten triggers → switch to late_ema
   - Late_ema flatten triggers (even if all gains negative) → switch to late_qat with `qat_lr_fraction:0.150`
   - QAT val_bpb holds steady or improves slightly
   - If QAT flattens → early stop fires (`cascade:early_stop ... reason:late_qat_plateau`)

5. If 10k validates well, scale to 20k:
   - Set `ITERATIONS=20000` and `ADAPTIVE_EVAL_HISTORY_PATH` to the 10k jsonl
   - Consider `WARMDOWN_ITERS=2400` (scale proportionally)

## Current strategic direction

- `adaptive-cascade-controller` is the active development branch
- Three critical cascade bugs now fixed; no validated end-to-end successful QAT cascade run yet
- `QAT_LR_FRACTION=0.15` calibrated from record-setting archive runs
- Next milestone: a clean 10k run where QAT either holds or improves int8 roundtrip val_bpb vs EMA-only
- `master` remains the stable training branch
- `feature/memory-optimizations` deferred (compute-bound, not memory-bound on current hardware)
