# 🧠 Parameter Golf — The Thinker

> **Strategy:** Maximum reasoning depth through recursive parameter tying, within a strict <16MB footprint.
> **Hardware:** Developed on Apple M1 Mac using MLX. Final runs on RunPod H100.

---

## What is this repo?

One of three variant forks of the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

**The challenge:** Build the best language model that fits in **16MB**, trains in **under 10 minutes on 8×H100 GPUs**, scored by bits-per-byte (bpb) on the FineWeb validation set. Lower bpb = smarter model. Baseline to beat: **1.2244 bpb**.

**The Thinker's bet:** You don't need more unique weights — you need more passes through the same weights.

---

## The Core Idea — Explained Simply

OpenAI's baseline has 9 *different* layers — 9 sets of unique weights, each doing something slightly different. Think of it as 9 different teachers each teaching one lesson.

The Thinker asks: what if you had **1 brilliant teacher** who teaches the **same lesson 30 times**, and the student gets smarter each pass?

That's **depth recurrence / parameter tying** — the core of this repo.

```
Standard baseline:
Input → Layer1 → Layer2 → Layer3 → ... → Layer9 → Output
         (9 unique weight matrices)

The Thinker:
Input → SharedLayer → SharedLayer → SharedLayer → ... (30×) → Output
         (1 weight matrix, reused — same 16MB, 30× the depth)
```

This is a form of **Universal Transformer** — an idea from academic research that has never been cleanly implemented within this challenge's constraints. OpenAI explicitly listed it in their wishlist.

---

## Why This Idea Is Genuinely Novel

Looking at the current leaderboard (top 24 entries as of March 25, 2026):

- Every top entry is a variation of the standard transformer
- Techniques like Int6 quantization, SmearGate, BigramHash are all incremental improvements
- **Zero entries use recursive / universal transformer architecture**

This is an open lane. If it works, it will be noticed.

---

## Architecture Details

### The three additions over baseline

**1. Shared weights (the main idea)**
Instead of `nn.ModuleList([Block(config) for _ in range(n_layer)])`, use a single `Block(config)` called N times in a loop.

**2. Step embedding (critical)**
Without this, the model is blind — it doesn't know if it's on pass 1 or pass 30. A learned step embedding is a tiny lookup table (one vector per step) added to the input at each pass. Costs almost no parameters, makes a large quality difference.

**3. ReLU² activation**
When the same layer runs 30 times, gradients can "die" — become zero — making the model stop learning. ReLU² (squaring the ReLU output) keeps gradients alive through deep recursion. This is not experimental — it's used in production models.

### Configuration targets

| Parameter | Baseline | The Thinker target |
|---|---|---|
| Unique layers | 9 | 1 |
| Recursive passes | 9 | 20–40 |
| Dims | 512 | 512 |
| Activation | GELU | ReLU² |
| Step embedding | None | Learned (N × dim) |
| Precision | Mixed | FP16 (room to spare) |

### Finding the optimal depth

There is a sweet spot. Too few passes = underpowered. Too many = training instability.

Planned ablation (all runnable on M1 Mac):

| Passes | Est. M1 time (200 steps) | Expected bpb |
|---|---|---|
| 10 | ~4 min | ~1.22 |
| 20 | ~6 min | ~1.20? |
| 30 | ~9 min | ~1.18? |
| 40 | ~12 min | TBD |

Run each, pick the best, then move to RunPod for full training.

---

## Development Setup — M1 Mac (Your Machine)

### One-time setup

```bash
# Clone your repo
git clone https://github.com/yuvaraj-97/parameter-golf-thinker.git
cd parameter-golf-thinker

# Create isolated Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install MLX and dependencies (M1-optimised)
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

### Download a tiny slice of data (do this first, once)

```bash
# Just 1 shard — enough for all local experiments (~800MB download)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

This downloads to `./data/datasets/fineweb10B_sp1024/`. Takes ~5 minutes on a decent connection.

### Run the baseline first (always do this before changing anything)

```bash
RUN_ID=baseline_m1 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

**Expected output on M1:**
- Runtime: ~3–5 minutes
- Final val_bpb: ~1.22–1.25
- If you see this number, your environment works perfectly

**Do not change any code until this runs clean.** This is your reference point.

---

## Implementing The Thinker — Step by Step

Open `train_gpt.py`. Find the `GPT` class. Make these changes:

### Change 1 — Replace the layer list with a single shared layer

**Before (find this in the file):**
```python
self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
```

**After (replace with this):**
```python
self.shared_block = Block(config)          # one block shared across all passes
self.n_recursive = config.n_layer          # how many times to reuse it
self.step_embed = nn.Embedding(            # tells model which pass it's on
    self.n_recursive, config.n_embd
)
```

### Change 2 — Replace the forward pass loop

**Before (find this in the forward method):**
```python
for block in self.layers:
    x = block(x)
```

**After (replace with this):**
```python
for step in range(self.n_recursive):
    step_signal = self.step_embed(
        torch.tensor(step, device=x.device)
    )
    x = x + step_signal          # inject "which pass am I on?"
    x = self.shared_block(x)     # run the shared layer
```

### Change 3 — Switch activation to ReLU²

Find where GELU is used inside the `MLP` class and replace:

**Before:**
```python
self.act = nn.GELU()
```

**After:**
```python
def relu_squared(x):
    return F.relu(x) ** 2
self.act = relu_squared
```

That's the full Thinker implementation. Three changes, ~10 lines total.

### Run the Thinker on M1

```bash
RUN_ID=thinker_30passes \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Compare the val_bpb to your baseline run. If it's lower, the idea is working.

---

## Scaling Up — RunPod (When M1 Experiments Look Promising)

Only move to RunPod once you've confirmed on M1 that:
- The code runs without crashing
- val_bpb is better than 1.2244 (even slightly)
- You know which depth (passes) works best

### RunPod setup

1. Create account at runpod.io
2. Use the official Parameter Golf template: `console.runpod.io/deploy?template=y5cejece4j`
3. Start with **1×A100** (~$2.50/hr) — not 8×H100 yet
4. SSH in and clone your repo
5. Run full training:

```bash
RUN_ID=thinker_full \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Only use 8×H100 ($20/hr) for the final leaderboard submission after everything is proven.

---

## Community Techniques to Stack on Top

Once recursive layers are proven, add these one at a time (each proven on the leaderboard):

| Technique | bpb gain | Effort | Add in order |
|---|---|---|---|
| 11 passes instead of 9 | ~0.01 | Trivial | First |
| Sliding window eval | ~0.02 | Easy | Second |
| Muon WD=0.04 | ~0.01 | Trivial | Third |
| Int6 quantization | ~0.03 | Medium | Fourth |
| EMA weight averaging | ~0.005 | Easy | Fifth |
| BigramHash | ~0.01 | Medium | Sixth |

Add one at a time. Measure after each. If a technique hurts score, remove it.

---

## Budget Plan

| Phase | Where | Cost |
|---|---|---|
| All development & iteration | M1 Mac | **$0** |
| Full training validation | 1×A100 RunPod (~4 hrs) | ~$10 |
| 3-seed statistical proof | 1×A100 RunPod (~6 hrs) | ~$15 |
| Final leaderboard run | 8×H100 RunPod (~1 hr) | ~$20 |
| **Total for The Thinker** | | **~$45** |

Your M1 Mac saves you roughly $30–40 compared to doing all development on cloud GPUs.

---

## Submission Checklist

Before opening a PR on the main OpenAI repo:

- [ ] val_bpb beats current SOTA by at least 0.005 nats
- [ ] 3 training runs completed (different seeds) showing consistent improvement
- [ ] `README.md` written explaining the recursive layer approach clearly
- [ ] `submission.json` filled with your name, GitHub ID, val_bpb
- [ ] `train.log` attached showing all 3 runs
- [ ] `train_gpt.py` runs cleanly from inside the records folder

---

## Compute Grant Status

| Grant | Amount | Status |
|---|---|---|
| Quick-start | $25 | ✅ Received |
| Development grant | $500 | 🎯 Apply after first PR |
| Advanced competitor | $1,000 | 🚀 Apply after leaderboard entry |

Reapply at: **openai.com/index/parameter-golf/#credit-form**
Include your PR link as evidence.

---

## OpenAI Job Target — Why This Approach Matters

OpenAI's Chief Research Officer said the core question is: *"Can you come up with creative ideas in a sandbox setting?"*

The Thinker directly answers that. Universal Transformers are a real academic concept (arxiv.org/abs/1807.03819) that no one has cleanly implemented in this challenge. That's a creative, principled, well-reasoned bet — not a hyperparameter tweak.

Fill out the **participant form** (separate from the compute grant):
**jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf**

Do this today, before you write any code.

---

## Resources

- [OpenAI Parameter Golf repo](https://github.com/openai/parameter-golf)
- [Live leaderboard](https://parameter-golf.github.io/)
- [Community techniques thread](https://github.com/openai/parameter-golf/issues/140)
- [Parameter Golf Field Guide](https://sameersegal.github.io/learn-parameter-golf/)
- [Universal Transformer paper](https://arxiv.org/abs/1807.03819)
- [MLX documentation](https://ml-explore.github.io/mlx/) — Apple's M1-optimised ML framework
- [OpenAI Discord](https://discord.com/invite/openai) — #parameter-golf-discussions
