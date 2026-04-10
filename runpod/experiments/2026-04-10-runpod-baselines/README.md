# Runpod Baselines - 2026-04-10

This folder captures the first end-to-end Runpod verification runs for `parameter-golf-thinker`.

## Runs

| Run | Branch | Hardware | Iterations | Train Batch Tokens | Step `val_bpb` | Final roundtrip `val_bpb` | Train time | Eval time | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `master_1gpu_200` | `master` | `1x RTX 2000 Ada` | 200 | 32768 | 2.3237 | 2.3739 | 45393 ms | 145865 ms | First successful single-GPU smoke run |
| `memory_1gpu_200` | `feature/memory-optimizations` | `1x RTX 2000 Ada` | 200 | 32768 | 2.3627 | 2.3999 | 104726 ms | 354486 ms | Lower VRAM, slower and worse than `master` |
| `master_8gpu_200` | `master` | `8x H100 80GB HBM3` | 200 | 262144 | 2.1647 | 2.2756 | 5887 ms | 2739 ms | Distributed smoke validation |
| `master_8gpu_5000` | `master` | `8x H100 80GB HBM3` | 5000 | 262144 | 1.4847 | 1.4932 | 147561 ms | 2655 ms | Strong scaling result |
| `master_8gpu_20000` | `master` | `8x H100 80GB HBM3` | 20000 | 262144 | 1.4589 | 1.4785 | 612766 ms | 2670 ms | Slightly above strict 10 min training budget |

## Telemetry Highlights

### 1x RTX 2000 Ada

- GPU utilization during active training: ~92%
- VRAM used: ~864 MiB / 16 GiB on the feature-branch smoke run
- Observed behavior: compute-bound, not memory-bound

### 8x H100 80GB HBM3

- GPU utilization during active training: 100%
- VRAM used: ~7 GiB / 80 GiB per GPU
- CPU load: ~4%
- Host RAM used: ~16.55 GiB / 1.83 TiB
- Disk used: ~1 GiB / 100 GiB
- Observed behavior: strongly compute-bound, substantial VRAM headroom left unused

## Conclusions So Far

1. `master` is the default branch for cloud runs.
2. `feature/memory-optimizations` reduces VRAM but loses on both speed and quality in the tested setup.
3. Current H100 runs are not memory-limited, so pure memory-saving changes are unlikely to help unless they unlock larger batch, longer context, or a larger model.
4. The 8x H100 path is fully validated for this repo on Runpod.

## Hypotheses To Test Next

### 1. Batch or model scaling on H100

Because VRAM usage is far below the hardware limit, test whether increasing capacity or work per step improves `val_bpb` before the 10-minute wallclock cutoff.

Candidate directions:

- larger `TRAIN_BATCH_TOKENS`
- larger model dims / layers if artifact size allows
- longer context if the implementation supports it cleanly

### 2. Two-stage training schedule

The 20k run improved only modestly over 5k. That suggests trying a hybrid schedule instead of simply running longer with one regime.

Example shape:

1. Train the base config for the first 5k iterations.
2. Switch to a second regime once the slope flattens.

Candidate second-stage changes:

- lower LR / different warmdown
- change eval or quantization behavior only late
- enable a more expensive method only after the fast early learning phase
- use a branch-specific optimization only in stage two if it slows down early training

This idea is plausible, but it still needs actual A/B testing against a plain 20k baseline.

## Suggested Next Experiments

1. Keep `master` and run a 10-minute constrained sweep over `TRAIN_BATCH_TOKENS`.
2. Test one two-stage schedule against the current 20k baseline.
3. Revisit memory-oriented ideas only if they unlock a bigger configuration on hardware that is actually memory-constrained.
