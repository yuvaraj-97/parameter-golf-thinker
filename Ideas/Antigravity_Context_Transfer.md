# Session Notes — The Thinker Development Log

## Session 1: April 8-9, 2026 (M1 Mac Local Testing)

### What We Built
The "Thinker" — a Universal Transformer variant for the [Parameter Golf Challenge](https://github.com/KellerJordan/modded-nanogpt).
Instead of 9 separate layers, we use **1 shared layer** looped recursively with learned step embeddings.

### Architecture Summary
| Parameter          | Value   |
|--------------------|---------|
| Model Dim          | 512     |
| Heads              | 8       |
| KV Heads           | 4       |
| Vocab Size         | 1024    |
| Seq Length          | 1024    |
| Parameters         | 2.36M   |
| Compressed Size    | ~1.07MB |
| 16MB Budget Used   | ~6.4%   |

### Experiment Results

#### Run 1: 9-pass (default), 200 iterations, wallclock capped
- **Training stopped at:** Step 113/200 (~10 min wallclock cap)
- **Final val_bpb:** `2.6515` (after INT8+zlib quantization roundtrip)
- **Training time:** ~8.5 minutes (training) + ~2 hours (validation on M1)
- **Observation:** Loss dropped from 6.94 → ~4.5 over 113 steps. Model is severely undertrained.
- **File size:** 1,072,920 bytes (1.07 MB) — massively under the 16MB limit.

#### Run 2: 30-pass, 200 iterations, wallclock disabled
- **Training completed:** All 200 iterations in ~24 minutes
- **Step speed:** ~4.2 seconds/step (vs 2.5s for 9-pass)
- **Observation:** Loss at step 10 was 5.42 (vs 6.19 for 9-pass at step 10) — 30 passes learns faster per step!
- **Validation:** Still computing (60,568 chunks at ~7.8s/chunk)

### Key Learnings

1. **30-pass is 3x slower per step** but learns more per step (lower loss at same step count)
2. **On 8x H100s**, the 10-min wallclock will likely allow ~8,000-11,000 iterations at 30 passes (vs 20,000 at 9 passes)
3. **Model is incredibly small** — only using 6.4% of the 16MB budget. This means we can massively increase `MODEL_DIM` or `NUM_HEADS` to pack more intelligence
4. **Batch size matters:** `TRAIN_BATCH_TOKENS` controls how many tokens are processed per learning step. Larger = smoother learning but more memory. Currently using 8,192 locally.

### Code Changes Made

1. **Commit `5a68dbe`:** Original Thinker architecture (shared block + step embeddings + ReLU²)
2. **tqdm integration:** Added progress bar to MLX training loop
3. **Commit `964640e` (branch: `feature/memory-optimizations`):**
   - Added `USE_CHECKPOINTING` env var (default=1)
   - PyTorch: `torch.utils.checkpoint` wraps `shared_block` during training
   - MLX: `mx.checkpoint` wraps `shared_block` during training
   - Reduces peak VRAM ~70% at cost of ~20-30% recompute

### Optimization Ideas (See `Ideas/01-memory-and-speed-optimizations.md`)
1. **Gradient Checkpointing** ✅ Implemented
2. **FP8 Mixed Precision** — H100 only, ~2x speed, minimal quality loss
3. **Reduce Sequence Length** — Fallback from 1024 → 512 if memory is still tight
4. **`torch.compile()` + FlashAttention** — Fuse GPU kernels for speed

### Next Steps
1. Compare 30-pass checkpointed run vs non-checkpointed (memory + speed)
2. If val_bpb is promising, scale up `MODEL_DIM` (we have 15MB of headroom!)
3. Run ablation: 10 vs 20 vs 30 vs 40 passes
4. Deploy to RunPod (8x H100) for full 20,000-iteration competitive run

### Commands Reference
```bash
# Local M1 Mac — 30 passes, checkpointing on, no wallclock limit
source .venv/bin/activate
RUN_ID=thinker_30p_checkpoint \
ITERATIONS=200 \
NUM_LAYERS=30 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=0 \
USE_CHECKPOINTING=1 \
python3 train_gpt_mlx.py
```
