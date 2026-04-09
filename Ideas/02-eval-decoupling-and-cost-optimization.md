# Eval Decoupling & Cost Optimization

This document captures ideas around separating the evaluation phase from training to reduce RunPod costs and M1 iteration time.

## The Problem

Current flow in `train_gpt_mlx.py` (and `train_gpt.py`):

1. Training loop runs (25 min on M1, ~10 min on H100)
2. Weights saved to `.npz` and `.int8.ptz` ← **already done here**
3. Final roundtrip eval: loads `.int8.ptz`, dequantizes, runs full validation pass

Step 3 is pure verification. On M1, it takes ~5.5 hours (60,568 chunks at ~7.8s/chunk).  
On RunPod 8x H100, it would be fast (~minutes), but **you're paying H100 rates while the GPU sits mostly idle doing sequential inference**.

The official challenge server runs its own evaluation on submission — so the local roundtrip eval is **optional overhead**, not a hard requirement.

## Key Insight

Weights are saved **before** the roundtrip eval runs. You can kill the process after training completes and the `.int8.ptz` is written — the submission artifact is already ready.

```
train_gpt_mlx.py lines 1062-1079: weights saved here ✅
train_gpt_mlx.py lines 1081-1097: roundtrip eval here ← optional
```

## Optimization Ideas

### 1. Skip Final Roundtrip Eval in Dev Runs (FREE WIN)

Add an env var `SKIP_ROUNDTRIP_EVAL` (default=0 for submission runs, =1 for dev):

```python
skip_roundtrip = bool(int(os.environ.get("SKIP_ROUNDTRIP_EVAL", "0")))
if not skip_roundtrip:
    # ... run roundtrip eval ...
```

This alone saves 5.5 hours on M1 and kills the RunPod meter as soon as training finishes.

### 2. Decouple Eval Into a Separate Script

Extract `eval_val` into a standalone `eval_model.py` that:
- Loads the `.int8.ptz` artifact
- Runs the full validation pass
- Prints `val_loss` and `val_bpb`

This allows:
- Train on expensive H100 → kill instance immediately after `.int8.ptz` is written
- Run eval locally on M1 overnight (free, no cost)
- Or run eval on a cheaper CPU/GPU instance (e.g., A10G at $0.75/hr vs H100 at $3.50/hr)

### 3. Cheaper Hardware for Local Eval

The eval is **inference-only** (no gradients, no optimizer). Hardware options:
- **M1 Mac**: free, but ~5.5 hrs for full validation
- **A10G (24GB)** on RunPod: ~$0.75/hr, ~10-15x faster than M1 → ~25 min eval
- **A100 (80GB)** on RunPod: ~$2.50/hr, ~30-40x faster than M1 → ~10 min eval
- **8x H100**: only needed for the official challenge submission run

For dev iteration, A10G is the sweet spot: cheap, fast enough to get real feedback.

### 4. VAL_LOSS_EVERY During Training (Quick Sanity Checks)

Instead of relying on the full final eval, set `VAL_LOSS_EVERY=500` or similar during training to catch divergence early. This uses a small subset of the validation set and is much faster per check.

The current default is `VAL_LOSS_EVERY=0` (disabled) locally, which means you only find out if training worked after the full 5.5hr eval completes.

### 5. Reduce Eval Batch Size for Faster Dev Feedback

`VAL_BATCH_SIZE` controls how many tokens are processed per chunk during eval.  
Increasing this (e.g., from 8192 to 65536) on a GPU would reduce the number of serial chunks from 60,568 to ~7,500, cutting eval time proportionally.

## Hardware Decision Matrix

| Scenario | Hardware | Cost | Time |
|---|---|---|---|
| Local dev iteration | M1 Mac | Free | 5.5 hrs (skip if possible) |
| Quick sanity check | A10G RunPod | ~$0.75/hr | ~25 min |
| Pre-submission validation | A100 RunPod | ~$2.50/hr | ~10 min |
| Official submission run | Challenge 8x H100 | Free (challenge server) | Minutes |

## Recommended Dev Loop

```bash
# Local M1: train + skip roundtrip eval
SKIP_ROUNDTRIP_EVAL=1 \
VAL_LOSS_EVERY=0 \
NUM_LAYERS=30 \
ITERATIONS=200 \
python3 train_gpt_mlx.py
# → weights in outputs/ after ~25 min, done

# RunPod A10G: validate the artifact (optional)
# python3 eval_model.py --model outputs/thinker_30p_mlx_model.int8.ptz
```

## Relationship to Idea 01

These optimizations are **orthogonal** to the memory/speed ideas in `01-memory-and-speed-optimizations.md`:
- Idea 01 targets: training speed (FP8, gradient checkpointing, batch tuning)
- Idea 02 targets: eval cost (decoupling, hardware selection, skipping unnecessary eval)

They can and should be combined in a "sandwich" branch that applies both.
