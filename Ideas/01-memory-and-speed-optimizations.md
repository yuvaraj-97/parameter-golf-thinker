# Future Optimization Ideas for The Thinker

This document serves as a prompt and reference file for future development sessions on the `parameter-golf-thinker` repository. Paste this file to your AI assistant when you resume work to immediately provide them with the necessary context.

## The Core Concept
The "Thinker" architecture relies on passing data through the exact **same 1 shared layer multiple times** (e.g., 30 recursive passes) using a "step embedding" to keep track of the depth. This maintains an incredibly small parameter footprint (~1.07MB, well under the 16.7MB challenge limit) while artificially increasing reasoning depth.

## The Primary Engineering Challenge
Processing a single batch with 30 passes takes roughly 3x longer than the standard 9-pass architecture. When running on the official 8x H100 evaluation server with a strict 10-minute boundary cutoff limit, a 30-pass model will likely complete significantly fewer iterations than the standard 20,000 steps.

## Required Memory and Speed Optimizations
To speed up iterations and allow the 30-pass model to train on more data within the 10-minute constraint without bottlenecking the system, implement the following optimizations in `train_gpt.py`:

1. **FP8 Mixed Precision (Hardware natively supported on H100):**
   The H100 GPUs support physical FP8 tensor cores. Using `FP8` precision for matrix multiplications instead of standard BF16 directly doubles mathematical computational speed and halves memory bandwidth usage, allowing drastically more iterations before the 10-minute clock stops.

2. **Gradient Checkpointing (Activation Checkpointing):**
   A 30-pass architecture keeps all 30 intermediate activations in VRAM during the forward pass to compute gradients later. This can instantly exhaust GPU memory.
   *Idea:* Implement PyTorch Gradient Checkpointing (Activation Checkpointing) to dump intermediate VRAM and re-calculate the passes exactly when needed during backpropagation.
   *Trade-off:* Freeing up this 70% VRAM constraint allows bumping the `TRAIN_BATCH_TOKENS` up massively for brutally powerful learning steps, at the cost of being roughly 20-30% slower per individual step.

3. **Reduce Sequence Length / Batch Size:**
   Experiment with reducing `TRAIN_BATCH_TOKENS` or shrinking sequence length (e.g. from 1024 to 512). This processes batches significantly faster, pushing the model to see more distinct iterations before the 10-minute bell.

4. **Optimize `torch.compile()` and FlashAttention:**
   When running the PyTorch equivalent on the H100, ensure that `F.scaled_dot_product_attention` is fully firing alongside `torch.compile(model)` to aggressively fuse GPU memory kernels.

## Instructions for Next Session
When modifying the code, review these exact techniques above. Focus first on seamlessly integrating **Gradient Checkpointing** to prevent OOM (Out Of Memory) issues when running 30 layer loops, immediately followed by configuring the matrices for **FP8** to literally cheat the physics of the H100 clock speed limits.
