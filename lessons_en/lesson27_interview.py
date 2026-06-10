"""
Lesson 27: LLM Interview Prep — High-Frequency Questions & Advanced Topics
==========================================================================

This lesson compiles the most frequently asked interview questions in the LLM field:
  1. Architecture — Transformer core mechanisms
  2. Training & Alignment — Pre-training / SFT / RLHF / DPO
  3. Inference Optimization — KV Cache / Quantization / Speculative Decoding
  4. Cutting-Edge Tech — MoE / Mamba / GRPO / Multimodal
  5. Engineering Practice — Distributed training / Deployment / Long context
  6. Math Derivations — Hand-derive core formulas

Each question includes: Question → Core Answer → Deep Follow-up → Code Verification

Run: python lessons_en/lesson27_interview.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# Part 1: Architecture
# ============================================================

print("=" * 70)
print("Part 1: Architecture — Transformer Core Mechanisms")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q1: Why Self-Attention instead of RNN?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  1. Parallelism: RNN must process sequentially (time t depends on t-1),
     Attention can compute all positions simultaneously
  2. Long-range dependencies: RNN gradients vanish over many steps,
     Attention has O(1) distance between any two positions
  3. Computational complexity:
     - RNN: O(n * d^2) — n is sequence length, d is dimension
     - Attention: O(n^2 * d) — quadratic in sequence, but parallelizable
     - When n < d^2/n (short sequences), Attention is faster

Deep Follow-up:
  Q: How to solve Attention's O(n^2) complexity?
  A: Multiple approaches:
     - Flash Attention: Doesn't reduce complexity, but reduces HBM access (IO-aware), 2-4x faster in practice
     - Sparse Attention: Only attend to subset of positions (Longformer, BigBird)
     - Linear Attention: Approximate softmax with kernel functions (Performers)
     - State Space Models: O(n) complexity (Mamba, S4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q2: Why Multi-Head Attention?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  Single-head Attention can only learn one "attention pattern" (e.g., only syntactic relations).
  Multi-head allows the model to simultaneously attend to different types of relations:
    - Head 1: Syntactic relations (subject -> predicate)
    - Head 2: Coreference (pronoun -> noun)
    - Head 3: Positional relations (adjacent words)
    - Head 4: Semantic relations (synonyms)

  Mathematically: MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_o
  Each head has dimension = d_model / n_heads, total computation unchanged

Deep Follow-up:
  Q: Why not just use one large head (e.g., single head with d=512)?
  A: Experiments show multi-head works better. Analogy: 8 experts each
     looking from one angle > 1 generalist looking at everything.
     A large head tends to learn "average" patterns; multi-head learns "diverse" patterns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q3: LayerNorm vs BatchNorm? Why does Transformer use LayerNorm?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  BatchNorm: Normalize along batch dimension -> each feature normalized across samples
  LayerNorm: Normalize along feature dimension -> each sample normalized internally

  Transformer uses LayerNorm because:
  1. Variable sequence length -> BatchNorm statistics are unstable
  2. Small batch size at inference (bs=1) -> BatchNorm degrades
  3. Autoregressive generation with growing sequence -> BatchNorm can't handle
  4. LayerNorm normalizes each token independently, unaffected by batch

  Further: RMSNorm vs LayerNorm
    LayerNorm: y = (x - mu) / sigma * gamma + beta  (subtract mean + divide std + scale + shift)
    RMSNorm:   y = x / RMS(x) * gamma               (only divide RMS + scale, no mean/beta)
    RMSNorm saves mean and bias computation, ~10% faster, similar effectiveness

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q4: Why do Residual Connections work?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  1. Gradient highway: During backprop, gradients can flow directly through the residual path
     y = x + F(x) -> dL/dx = dL/dy * (1 + dF/dx)
     Even if dF/dx is small, the 1 ensures gradients don't vanish

  2. Information preservation: Each layer only needs to learn the "increment" (residual)
     Analogy: Editing a draft (residual) vs rewriting from scratch (no residual)

  3. Identity initialization: If F(x)=0, y=x, at least it's no worse than a shallower network

Deep Follow-up:
  Q: Pre-Norm vs Post-Norm?
  A: Pre-Norm (LayerNorm before Attention) is more stable, no warmup needed
     Post-Norm (LayerNorm after Attention) may achieve better results but is unstable
     Mainstream uses Pre-Norm (GPT, LLaMA all use Pre-Norm)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q5: Why is positional encoding important? What is RoPE's core idea?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  Self-Attention is position-agnostic (permutation invariant):
    Attention(["I", "love", "you"]) = Attention(["you", "love", "I"])
  So positional information must be injected.

  RoPE (Rotary Position Encoding) core idea:
    Use rotation matrices to encode relative position, so q.k dot product
    depends only on relative distance

  How: Pair adjacent dimensions of q and k, rotation angle = position * frequency
    q_m = [q_0 cos(m*theta_0) - q_1 sin(m*theta_0), q_0 sin(m*theta_0) + q_1 cos(m*theta_0), ...]
    k_n = [k_0 cos(n*theta_0) - k_1 sin(n*theta_0), k_0 sin(n*theta_0) + k_1 cos(n*theta_0), ...]
    q_m . k_n = f(m-n)  <- depends only on relative position m-n!

  Why different frequencies?
    Low frequency (small theta): Slow rotation, captures long-range positional relations
    High frequency (large theta): Fast rotation, captures short-range positional relations
    Clock analogy: Second hand (high freq) for second-level, hour hand (low freq) for hour-level
""")

# --- Code Verification: RoPE relative position ---
print("\n[Code Verification] RoPE makes Attention depend only on relative position")
print("-" * 50)

d = 4
theta = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))

def apply_rope(x, pos):
    """Apply RoPE to vector x at position pos"""
    x_pairs = x.float().view(-1, 2)
    cos_val = torch.cos(pos * theta)
    sin_val = torch.sin(pos * theta)
    rotated = torch.stack([
        x_pairs[:, 0] * cos_val - x_pairs[:, 1] * sin_val,
        x_pairs[:, 0] * sin_val + x_pairs[:, 1] * cos_val
    ], dim=-1)
    return rotated.view(-1)

q = torch.randn(d)
k = torch.randn(d)

for m in range(4):
    for n in range(4):
        dot = torch.dot(apply_rope(q, m), apply_rope(k, n))
        if m - n == 1:
            print(f"  Position({m},{n}) relative_dist={m-n}: dot_product={dot.item():.4f}")

print("  -> Same relative distance = same dot product (RoPE's core property)")


# ============================================================
# Part 2: Training & Alignment
# ============================================================

print("\n" + "=" * 70)
print("Part 2: Training & Alignment — Pre-training / SFT / RLHF / DPO")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q6: What do the three stages of LLM training do?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  Stage 1 — Pre-training:
    Goal: Learn statistical patterns of language
    Data: Massive unlabeled text (web pages, books, code...)
    Task: Next token prediction
    Loss: CrossEntropyLoss
    Analogy: Read all of Wikipedia, learn to "speak"

  Stage 2 — Supervised Fine-Tuning (SFT):
    Goal: Learn to answer questions following instructions
    Data: Instruction-response pairs ("Explain quantum mechanics" -> "QM is...")
    Task: Conditional text generation
    Analogy: After reading Wikipedia, take "Q&A classes" to learn answering

  Stage 3 — Alignment (RLHF / DPO):
    Goal: Make responses align with human preferences (helpful, safe, honest)
    Data: Human preference annotations (response A is better than B)
    Methods: RLHF (train reward model + PPO) or DPO (directly optimize preferences)
    Analogy: Learn to "say good things" rather than "say random things"

Deep Follow-up:
  Q: Why can't we skip pre-training and go straight to SFT?
  A: SFT data is too small (tens of thousands vs hundreds of billions of tokens),
     the model can't learn language foundations.
     Analogy: Going to high school without elementary school — you won't understand.

  Q: Why is the alignment stage needed?
  A: SFT only teaches "how to answer", not "what answers are good".
     The model might learn to mimic format but produce harmful/useless content.
     Alignment teaches the model to distinguish "good" from "bad" responses.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q7: DPO vs RLHF — differences and trade-offs?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  RLHF pipeline:
    1. Train reward model RM (learn human preferences)
    2. Use PPO to optimize policy model (maximize RM reward)
    Requires: 4 models (policy, reference, reward, value)
    Problems: Complex training, unstable, many hyperparameters

  DPO pipeline:
    Directly optimize policy model with preference data, skip reward model
    Derivation:
      RLHF optimal: pi*(y|x) ~ pi_ref(y|x) * exp(r(x,y)/beta)
      Invert reward: r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))
      Substitute into Bradley-Terry model to get DPO loss:
      L_DPO = -log sigma(beta * [log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))])

  Comparison:
    +------------+--------------+--------------+
    |            |    RLHF      |     DPO      |
    +------------+--------------+--------------+
    | Models     | 4            | 2            |
    | Complexity | High         | Low          |
    | Stability  | Unstable     | Stable       |
    | Reward ML  | Required     | Not needed   |
    | Online     | Supported    | Not supported|
    | Ceiling    | Higher(theory)| Slightly lower|
    +------------+--------------+--------------+

Deep Follow-up:
  Q: When to use DPO vs RLHF?
  A: Limited resources / fast iteration -> DPO; Maximum performance / online data -> RLHF
     Many teams use DPO first for quick validation, then RLHF for fine-tuning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q8: What is GRPO? How does it differ from PPO?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  GRPO = Group Relative Policy Optimization
  A simplified RLHF method proposed by DeepSeek.

  Core idea: No value model needed; use "within-group relative ranking" instead of "absolute reward"

  PPO advantage: A(s,a) = Q(s,a) - V(s)  <- needs trained value model V
  GRPO advantage: A_i = (r_i - mean(r)) / std(r)  <- only needs group statistics

  Procedure:
    1. For each question, sample G responses
    2. Score each response with reward model: r_1, ..., r_G
    3. Compute group-normalized advantage: A_i = (r_i - mean) / std
    4. Update policy weighted by advantage

  Advantage: Eliminates value model, simpler training. Used in DeepSeek-V2/V3.
""")

# --- Code Verification: GRPO advantage ---
print("\n[Code Verification] GRPO Advantage Calculation")
print("-" * 50)

rewards = torch.tensor([[0.8, 0.3, 0.5, 0.1],
                         [0.9, 0.7, 0.2, 0.4]])
mean = rewards.mean(dim=1, keepdim=True)
std = rewards.std(dim=1, keepdim=True) + 1e-8
advantages = (rewards - mean) / std

print(f"Rewards: {rewards.tolist()}")
print(f"Group mean: {mean.squeeze().tolist()}")
print(f"Group std: {std.squeeze().tolist()}")
print(f"GRPO advantages: {[[f'{a:.2f}' for a in row] for row in advantages.tolist()]}")
print("-> Positive = better than group average, Negative = worse")


# ============================================================
# Part 3: Inference Optimization
# ============================================================

print("\n" + "=" * 70)
print("Part 3: Inference Optimization — KV Cache / Quantization / Speculative Decoding")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q9: What is KV Cache? Why does it speed up inference?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  During autoregressive generation, each step computes Attention(q_t, K, V)
  where K=[k_1,...,k_t], V=[v_1,...,v_t] contains info from all previous tokens

  Without KV Cache: Recompute all k_1,...,k_t every step (redundant!)
  With KV Cache: Only compute new k_t, v_t, append to cache

  Complexity comparison (generating T tokens):
    No Cache: O(T^2 * d)  — recompute all each step
    With Cache: O(T * d)  — only compute new each step

  Memory usage (LLaMA-2-70B example):
    Per-token KV Cache = 2 * n_kv_heads * d_head * seq_len * 2 bytes
    MHA (64 KV heads): ~32 KB/token -> 128MB for 4096 tokens
    GQA (8 KV heads):  ~4 KB/token  -> 16MB for 4096 tokens

Deep Follow-up:
  Q: How to solve KV Cache memory bottleneck?
  A: Multiple approaches:
     - GQA: Reduce KV heads (LLaMA-2 uses 8 instead of 64)
     - MQA: All heads share 1 set of KV (extreme GQA)
     - PagedAttention: Virtual memory management, avoid fragmentation (vLLM)
     - KV Cache quantization: FP16 -> INT8/INT4, save half/quarter memory
     - Sliding window: Only keep recent W tokens' KV (Mistral)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q10: What is model quantization? INT8/INT4 quantization?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  Quantization = Represent model parameters with fewer bits
  FP32 -> FP16 -> INT8 -> INT4

  Two quantization approaches:
  1. Post-Training Quantization (PTQ):
     Directly convert trained model parameters from FP16 to INT8/INT4
     Methods: AbsMax (divide by max absolute value), MinMax (map to [-128,127])

  2. Quantization-Aware Training (QAT):
     Simulate quantization error during training, let model adapt to low precision
     Better results but higher training cost

  Accuracy loss:
    FP16 -> INT8: Nearly lossless (<1% accuracy drop)
    FP16 -> INT4: Slight loss (1-3%), needs GPTQ/AWQ methods to compensate

  Memory and speed benefits:
    7B model FP16: 14GB VRAM
    7B model INT8: 7GB VRAM
    7B model INT4: 3.5GB VRAM (can run on a 6GB GPU!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q11: What is Speculative Decoding?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  Core tension: Large models produce high quality but are slow; small models are fast but lower quality
  Speculative Decoding: Small model "guesses", large model "verifies"

  Process:
    1. Small model (draft model) quickly generates K tokens
    2. Large model (target model) verifies all K tokens in one forward pass
    3. Accept correct tokens, reject incorrect ones, regenerate from rejection point

  Why does it speed up?
    - Large model can verify K tokens in one forward pass
    - If small model accuracy is p, expected acceptance length = 1/(1-p)
    - Small model accuracy is typically 70-90%, so expected 3-10 tokens accepted
    - Speedup: 2-3x (lossless — output is identical to pure large model)

Deep Follow-up:
  Q: Why is the output identical to the pure large model?
  A: On rejection, resample from the large model's distribution. Mathematically
     equivalent to sampling directly from the large model.
     This is the elegance of speculative decoding: speed without quality loss.
""")

# --- Code Verification: Quantization ---
print("\n[Code Verification] Quantization impact on precision")
print("-" * 50)

def quantize_int8(tensor):
    """Simple AbsMax INT8 quantization"""
    scale = tensor.abs().max() / 127
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale

def dequantize_int8(quantized, scale):
    return quantized.float() * scale

original = torch.randn(100)
q8, scale = quantize_int8(original)
recovered = dequantize_int8(q8, scale)
error = (original - recovered).abs().mean()

print(f"Original range: [{original.min():.4f}, {original.max():.4f}]")
print(f"INT8 quantized range: [{q8.min()}, {q8.max()}]")
print(f"Dequantized range: [{recovered.min():.4f}, {recovered.max():.4f}]")
print(f"Mean error: {error:.6f} (relative: {error/original.abs().mean()*100:.2f}%)")


# ============================================================
# Part 4: Cutting-Edge Tech
# ============================================================

print("\n" + "=" * 70)
print("Part 4: Cutting-Edge Tech — MoE / Mamba / GRPO / Multimodal")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q12: What is MoE (Mixture of Experts)? Pros and cons?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  MoE = Multiple "expert" networks replace a single FFN

  Structure:
    Input x -> Router (gating network) -> Select Top-K experts -> Weighted sum
    Router(x) = softmax(W_r * x)  -> weight for each expert
    output = sum gate_i * Expert_i(x)  (only Top-K selected)

  Advantages:
    1. Large params but small compute: 8 experts, activate 2, 8x params but same FLOPs
    2. Large model capacity: More params = more knowledge storage
    3. Specialization: Different experts learn different domain knowledge

  Problems:
    1. Router Collapse: All tokens routed to a few experts
       Solution: Add auxiliary loss aux_loss = alpha * n * sum(p_i^2) (encourage uniform)
    2. Load imbalance: Uneven expert distribution across GPUs
       Solution: Expert Parallelism + Capacity Factor
    3. Communication overhead: Cross-GPU All-to-All communication
       Solution: Reduce communication frequency, faster networks

  Representative models: Mixtral 8x7B, DeepSeek-V2/V3, Switch Transformer

Deep Follow-up:
  Q: Do MoE "experts" really learn different domain knowledge?
  A: Research shows experts do have some specialization, but not as clearly as
     "math expert" or "code expert" as one might expect.
     More like "shallow pattern" differentiation: some experts handle punctuation,
     others handle specific syntactic structures.
     Full domain specialization requires stronger constraints (e.g., expert selection loss).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q13: Mamba/SSM vs Transformer — differences?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  SSM (State Space Model) core equations:
    h'(t) = A*h(t) + B*x(t)   (state update)
    y(t)  = C*h(t) + D*x(t)   (output)

  Analogy:
    Attention = Meeting where everyone can talk directly to everyone (rich info but O(n^2))
    SSM = Meeting where each person only hears the previous person's summary (fast O(n) but compressed)
    Mamba = SSM + Selective mechanism: remember more for important info, less for unimportant

  Mamba's key innovation — Selective mechanism:
    Traditional SSM: A, B, C are fixed (same compression regardless of input)
    Mamba: A, B, C depend on input x (preserve more for important, compress more for unimportant)
    Analogy: When reading, take detailed notes on important paragraphs, only keywords for unimportant ones

  Comparison:
    +----------------+----------------+----------------+
    |                |  Transformer   |    Mamba       |
    +----------------+----------------+----------------+
    | Training       | O(n^2)         | O(n)           |
    | Inference      | O(n) per step  | O(1) per step  |
    | Long sequences | Limited (KV)   | Natural support|
    | Parallel train | Good           | Good (parallel)|
    | Effectiveness  | Mature/verified | Rapidly catching up|
    +----------------+----------------+----------------+

Deep Follow-up:
  Q: Will Mamba replace Transformer?
  A: Not in the short term. Transformer ecosystem is too mature, Mamba is still being validated.
     But hybrid architectures (Jamba = Mamba + Attention) may be the future.
     Long-sequence scenarios (DNA, audio) Mamba has natural advantages.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q14: How do multimodal LLMs (VLMs) align images and text?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  Core challenge: Images are 2D pixels, text is 1D tokens — how to make them "speak the same language"?

  Mainstream approach: Vision Encoder + Projection Layer + Language Model
    1. Vision Encoder (e.g., ViT/CLIP): Slice image into patches, extract visual features
       Image -> [patch_1, patch_2, ..., patch_N] -> visual features [v_1, ..., v_N]
    2. Projection Layer: Map visual features to language model embedding space
       v_i -> p_i = W_project * v_i  (dimension alignment)
    3. Language Model: Concatenate projected visual tokens with text tokens
       Input = [visual_token_1, ..., visual_token_N, text_token_1, ...]

  Representative models:
    - LLaVA: CLIP ViT + MLP projection + LLaMA
    - Qwen-VL: ViT + Cross-Attention + Qwen
    - GPT-4V: Architecture not public, likely similar approach

  How does PatchEmbedding slice images into tokens?
    1. Image 224x224x3 -> cut into 16x16 patches -> 14x14=196 patches
    2. Each patch 16x16x3=768 dims -> linear projection to d_model dims
    3. Add positional encoding -> 196 visual tokens

Deep Follow-up:
  Q: Why not use pixels directly as tokens?
  A: 224x224=50176 pixels, sequence too long (Attention O(n^2) can't handle it).
     After patching, only 196 tokens, manageable.
     Also, patch-level features have more semantic information than pixel-level.
""")


# ============================================================
# Part 5: Engineering Practice
# ============================================================

print("\n" + "=" * 70)
print("Part 5: Engineering Practice — Distributed Training / Deployment / Long Context")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q15: What are the distributed training parallelism strategies?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  1. Data Parallelism (DP):
     Each GPU has a full model copy, data is sharded
     Forward -> gather gradients -> average -> update
     Bottleneck: Gradient communication; model too large for single GPU

  2. ZeRO (Zero Redundancy Optimizer):
     Optimized DP, shard optimizer states/gradients/parameters
     ZeRO-1: Shard optimizer states -> 4x memory savings
     ZeRO-2: Shard optimizer + gradients -> 8x memory savings
     ZeRO-3: Shard optimizer + gradients + parameters -> Nx savings (N=GPU count)

  3. Tensor Parallelism (TP):
     Split a single matrix multiplication across GPUs
     Y = X * W -> column split W=[W1, W2], Y = [X*W1, X*W2]
     High communication, suitable within node (NVLink)

  4. Pipeline Parallelism (PP):
     Split model layers across GPUs
     GPU0: Layer 0-7, GPU1: Layer 8-15, ...
     Problem: Bubbles (GPU idle waiting)
     Solution: Micro-batches to fill the pipeline

  5. Sequence Parallelism (SP):
     Split long sequences across GPUs
     Suitable for ultra-long context (128K+ tokens)

  Practical combination (LLaMA-2-70B training example):
    TP=4 (intra-node 4-GPU tensor parallel)
    x PP=2 (2 pipeline stages)
    x DP=64 (64-way data parallel)
    = 512 A100 GPUs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q16: Long context — what are the technical approaches?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  1. Training-time context extension:
     - Position Interpolation: Compress positional encoding
     - YaRN: Handle by frequency groups (high freq extrapolate, low freq interpolate)
     - NTK-aware scaling: Adjust RoPE base frequency
     - Progressive extension: Train 4K first, extend to 32K, then 128K

  2. Inference-time optimization:
     - Sliding window attention: Only attend to recent W tokens
     - Sparse attention: Full attention for key tokens, local for others
     - KV Cache compression: Drop unimportant KV (e.g., H2O)
     - Chunked prefill: Process long input in batches

  3. RAG (Retrieval-Augmented Generation):
     Don't feed long docs directly; retrieve relevant passages first
     Good for: Knowledge-intensive tasks (QA, summarization)
     Not for: Tasks requiring full-text understanding (long-doc reasoning)

  Representative context lengths:
    GPT-4: 128K tokens
    Claude-3: 200K tokens
    Gemini-1.5: 1M-2M tokens
    LLaMA-3: 8K -> 128K (extended training)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q17: Model deployment options? How to choose?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Answer:
  Inference framework comparison:
    +-----------+----------------+----------------+----------------+
    |           |   vLLM         |  TensorRT-LLM  |   Ollama       |
    +-----------+----------------+----------------+----------------+
    | Speed     | Fast           | Fastest        | Medium         |
    | Ease      | Medium         | Complex        | Easiest        |
    | Quant     | Good           | Good           | Good           |
    | KV Cache  | PagedAttn      | Optimized      | Basic          |
    | Use case  | Production     | Max perf       | Local dev      |
    +-----------+----------------+----------------+----------------+

  Deployment strategies:
    - 7B model + consumer GPU (24GB): INT4 quantization + vLLM
    - 7B model + CPU only: GGUF quantization + llama.cpp
    - 70B model + multi-GPU: TP=4 + vLLM
    - Local dev/testing: Ollama (one command to start)
""")

# --- Code Verification: Parameter count ---
print("\n[Code Verification] Parameter count for different model sizes")
print("-" * 50)

def calc_params(vocab_size, d_model, n_layers, n_heads, n_kv_heads=None, ffn_mult=4, moe_experts=1, moe_topk=1):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    d_head = d_model // n_heads

    emb_params = vocab_size * d_model
    q_params = d_model * (n_heads * d_head)
    k_params = d_model * (n_kv_heads * d_head)
    v_params = d_model * (n_kv_heads * d_head)
    o_params = (n_heads * d_head) * d_model

    ffn_dim = int(ffn_mult * d_model * 2 / 3)
    ffn_dim = ((ffn_dim + 63) // 64) * 64
    if moe_experts == 1:
        ffn_params = 3 * d_model * ffn_dim
    else:
        ffn_params = moe_experts * 3 * d_model * ffn_dim
        ffn_params += d_model * moe_experts

    norm_params = 2 * d_model
    per_layer = q_params + k_params + v_params + o_params + ffn_params + norm_params
    total = emb_params + n_layers * per_layer + d_model
    return total

models = [
    ("MiniMind (26M)",  dict(vocab_size=6400, d_model=512, n_layers=8, n_heads=8, n_kv_heads=2)),
    ("LLaMA-2-7B",     dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=32)),
    ("LLaMA-2-70B",    dict(vocab_size=32000, d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8)),
    ("Mixtral 8x7B",   dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, moe_experts=8)),
]

for name, kwargs in models:
    params = calc_params(**kwargs)
    print(f"  {name}: {params/1e6:.1f}M params ({params/1e9:.2f}B)")


# ============================================================
# Part 6: Math Derivations
# ============================================================

print("\n" + "=" * 70)
print("Part 6: Math Derivations — Hand-Derive Core Formulas")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q18: Derive Self-Attention computation step by step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input: X in R^{n x d}  (n tokens, d-dim embedding)

  Step 1: Linear projection
    Q = X * W_Q    K = X * W_K    V = X * W_V
    Q, K, V in R^{n x d_k}

  Step 2: Compute attention scores
    S = Q * K^T / sqrt(d_k)    in R^{n x n}
    S[i][j] = q_i . k_j / sqrt(d_k)  (how much token i attends to token j)

  Step 3: Softmax normalization
    A[i][j] = exp(S[i][j]) / sum_k exp(S[i][k])
    Each row sums to 1, representing token i's attention distribution

  Step 4: Weighted sum
    Output = A * V    in R^{n x d_k}
    output_i = sum_j A[i][j] * v_j  (weighted average of all values)

  Why divide by sqrt(d_k)?
    When d_k is large, q.k has high variance, softmax output approaches one-hot
    Dividing by sqrt(d_k) normalizes variance, softmax output is smoother, better gradients

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q19: Derive DPO loss function step by step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1: RLHF objective — maximize reward while staying close to reference model
    max_pi E_{x,y~pi}[r(x,y)] - beta * KL(pi || pi_ref)

  Step 2: Closed-form solution (via Lagrangian duality)
    pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)
    where Z(x) = sum_y pi_ref(y|x) * exp(r(x,y)/beta) is the partition function

  Step 3: Invert the reward function (key step!)
    pi*(y|x) / pi_ref(y|x) = exp(r(x,y)/beta) / Z(x)
    log(pi*(y|x) / pi_ref(y|x)) = r(x,y)/beta - log Z(x)
    r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)

  Step 4: Substitute into Bradley-Terry preference model
    P(y_w > y_l | x) = sigma(r(x,y_w) - r(x,y_l))

    r(x,y_w) - r(x,y_l)
    = beta * [log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))]
    (Z(x) cancels! This is why DPO doesn't need a reward model)

  Step 5: Final DPO loss
    L_DPO = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))]

  Intuition: Increase probability of chosen response, decrease rejected, while staying close to reference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q20: Derive why low-rank is sufficient for LoRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Original weight: W in R^{d x d}, params = d^2
  LoRA decomposition: dW = A * B, A in R^{d x r}, B in R^{r x d}, params = 2dr

  Compression ratio: d^2 / (2dr) = d/(2r)
  When d=4096, r=8: ratio = 256x, only 0.4% of parameters

  Why is low-rank sufficient?
  1. Empirical finding: dW during fine-tuning has effective rank typically 8-64
     Effective rank = number of "significant" singular values in dW
     This means fine-tuning only changes 8-64 directions, others barely change

  2. Theoretical explanation:
     Pre-trained model has already found a good position in high-dimensional space
     Fine-tuning only needs small adjustments near this position
     Analogy: No need to tear down walls when renovating, painting and curtains suffice

  3. SVD perspective:
     dW = U*S*V^T, most singular values are small
     Keep only top r largest: dW ~ U_r*S_r*V_r^T
     This is LoRA's A*B (A=U_r*sqrt(S_r), B=sqrt(S_r)*V_r^T)
""")

# --- Code Verification: LoRA low-rank effectiveness ---
print("\n[Code Verification] LoRA low-rank effectiveness: simulating fine-tuning weight changes")
print("-" * 50)

d = 512
W_pretrained = torch.randn(d, d)

true_rank = 8
A_true = torch.randn(d, true_rank) * 0.01
B_true = torch.randn(true_rank, d) * 0.01
delta_W = A_true @ B_true

U, S, Vh = torch.linalg.svd(delta_W)
print(f"Weight change dW shape: {delta_W.shape}")
print(f"Top 10 singular values: {S[:10].tolist()}")
print(f"Singular value decay ratio (10th/1st): {(S[9]/S[0]).item():.6f}")
print(f"Effective rank (singular values > 1% of max): {(S > S[0]*0.01).sum().item()}")
print("-> Most singular values near zero, low-rank approximation suffices")


# ============================================================
# Part 7: Common Pitfalls & Recent Advances
# ============================================================

print("\n" + "=" * 70)
print("Part 7: Common Pitfalls & Recent Advances")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q21: Common misconceptions clarified
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GQA vs MQA vs MHA:
   MHA: Each head has independent K,V (n_kv_heads = n_heads)
   GQA: K,V heads < Q heads, multiple Q heads share one KV group
   MQA: All Q heads share 1 KV group (n_kv_heads = 1, extreme GQA)

2. Temperature vs Top-K vs Top-P:
   Temperature: Controls distribution "sharpness" (T down -> more certain, T up -> more random)
   Top-K: Only sample from top K highest-probability tokens
   Top-P: Only sample from smallest token set whose cumulative probability >= P
   Can be combined: Temperature scaling -> Top-K truncation -> Top-P filtering

3. Pre-training Loss vs SFT Loss:
   Pre-training loss measures "next token prediction ability"
   SFT loss measures "instruction-following ability"
   Low pre-training loss != good at answering questions (needs SFT alignment)

4. Embedding vs Hidden State:
   Embedding: Word-to-vector mapping (lookup operation, nn.Embedding)
   Hidden State: Representation after Transformer layer processing
   Embedding is "initial understanding", Hidden State is "deep understanding"

5. Cross-Entropy vs KL Divergence:
   Cross-Entropy: H(p, q) = -sum p(x) log q(x)  (gap between prediction q and truth p)
   KL Divergence: KL(p||q) = H(p,q) - H(p)       (extra entropy term)
   When p is one-hot (like labels), CE = KL (since H(p)=0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q22: 2024-2025 LLM New Features Overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DeepSeek-V3 (2024.12):
   - MoE + MLA (Multi-head Latent Attention, new KV Cache compression)
   - FP8 mixed-precision training (first large-scale validation of FP8 training)
   - GRPO reinforcement learning (eliminates value model)
   - 671B params, 37B active (only 5.5% of params per token)

2. MLA (Multi-head Latent Attention):
   - Core idea: Compress KV into low-dimensional latent space
   - KV Cache stores compressed c_kv (much smaller than d_model)
   - Decompress back to k,v at inference time
   - Goes further than GQA: GQA reduces head count, MLA reduces dimensions

3. LLaMA 3 (2024.4):
   - 8B/70B/405B three sizes
   - 128K context (via long context extension training)
   - GQA (both 8B and 70B use GQA)
   - Training data: 15T tokens

4. Mamba-2 (2024.5):
   - Unified SSM + Attention framework (SSD, Structured State Space Duality)
   - 2-8x faster than Mamba-1
   - Proves SSM and Attention are special cases of the same mathematical framework

5. New quantization methods:
   - GPTQ: Post-training quantization based on approximate second-order information
   - AWQ: Activation-aware weight quantization (protect important weights)
   - GGUF: llama.cpp quantization format, supports CPU inference
   - FP8 training: First large-scale validation by DeepSeek-V3

6. Inference frameworks:
   - vLLM: PagedAttention, high-throughput inference
   - SGLang: Programmatic LLM invocation, RadixAttention
   - TensorRT-LLM: NVIDIA official, maximum performance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q23: Must-know coding questions for interviews
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Implement Self-Attention from scratch (most frequently asked)
2. Implement Multi-Head Attention
3. Implement RoPE positional encoding
4. Implement SwiGLU FFN
5. Implement RMSNorm
6. Implement GQA
7. Implement KV Cache + autoregressive generation
8. Implement DPO Loss
9. Implement LoRA Linear layer
10. Implement a simple MoE layer
""")

# --- Coding demos ---
print("\n[Must-Know Code] Self-Attention")
print("-" * 50)

def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    return attn @ V

n, d, d_k = 4, 8, 8
X = torch.randn(n, d)
W_q = torch.randn(d, d_k)
W_k = torch.randn(d, d_k)
W_v = torch.randn(d, d_k)
output = self_attention(X, W_q, W_k, W_v)
print(f"Input: {X.shape} -> Output: {output.shape}")
print(f"First 2 tokens: {output[:2, :4].detach().tolist()}")

print("\n[Must-Know Code] RoPE")
print("-" * 50)

def apply_rope(x, seq_len, dim):
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, dim/2)
    cos_val = torch.cos(angles)  # (seq_len, dim/2)
    sin_val = torch.sin(angles)  # (seq_len, dim/2)
    # Reshape for broadcasting: (1, seq_len, 1, dim/2)
    cos_val = cos_val.unsqueeze(0).unsqueeze(2)
    sin_val = sin_val.unsqueeze(0).unsqueeze(2)

    x1, x2 = x[..., ::2], x[..., 1::2]  # (1, seq_len, n_heads, dim/2)
    out_x1 = x1 * cos_val - x2 * sin_val
    out_x2 = x1 * sin_val + x2 * cos_val
    return torch.stack([out_x1, out_x2], dim=-1).flatten(-2)

x_test = torch.randn(1, 8, 2, 16)
x_rope = apply_rope(x_test, 8, 16)
print(f"RoPE input: {x_test.shape} -> output: {x_rope.shape}")
print("-> Adjacent dimensions paired and rotated, encoding position info")

print("\n[Must-Know Code] DPO Loss")
print("-" * 50)

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return loss.mean()

p_chosen = torch.tensor([-1.2, -0.8, -1.5])
p_rejected = torch.tensor([-2.1, -1.9, -2.3])
r_chosen = torch.tensor([-1.0, -0.7, -1.3])
r_rejected = torch.tensor([-1.8, -1.6, -2.0])

loss = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected)
print(f"DPO Loss: {loss.item():.4f}")
print("-> Makes chosen reward higher than rejected (logsigmoid ensures monotonicity)")

print("\n[Must-Know Code] LoRA Linear")
print("-" * 50)

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

lora_layer = LoRALinear(512, 512, rank=8)
base_params = sum(p.numel() for p in lora_layer.linear.parameters())
lora_params = sum(p.numel() for p in [lora_layer.lora_A, lora_layer.lora_B])
print(f"Base params: {base_params:,}")
print(f"LoRA params: {lora_params:,} ({lora_params/base_params*100:.1f}%)")
print(f"Compression ratio: {base_params/lora_params:.0f}x")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("Interview Preparation Roadmap")
print("=" * 70)

print("""
+-------------------------------------------------------------+
| Level 1 — Must Know (Junior)                                |
|  * Self-Attention principles and computation                |
|  * Transformer overall architecture                         |
|  * Pre-training / SFT / RLHF three stages                  |
|  * KV Cache principles                                      |
|  * Quantization basics                                      |
|  * Implement Self-Attention from scratch                    |
+-------------------------------------------------------------+
| Level 2 — Deep Understanding (Mid-level)                    |
|  * RoPE positional encoding                                 |
|  * GQA / MQA / MHA differences                             |
|  * DPO vs RLHF derivation                                  |
|  * LoRA principles and implementation                       |
|  * MoE architecture and load balancing                      |
|  * Distributed training strategies                          |
|  * Implement RoPE / DPO Loss / LoRA from scratch            |
+-------------------------------------------------------------+
| Level 3 — Cutting-Edge Mastery (Senior)                     |
|  * Mamba / SSM principles                                   |
|  * MLA (Multi-head Latent Attention)                        |
|  * GRPO training pipeline                                   |
|  * Speculative decoding                                     |
|  * Long context extension techniques                        |
|  * Multimodal architectures                                 |
|  * FP8 training                                             |
|  * Implement MoE / KV Cache + generation from scratch       |
+-------------------------------------------------------------+

Interview tips:
  1. Start with intuitive analogy, then math formulas (shows depth of understanding)
  2. Proactively mention "I can also elaborate on XXX" (guide interviewer to your strengths)
  3. If unsure, say "As I understand it... but I'd need to verify details" (honesty > guessing)
  4. For coding questions, write the framework first then fill details (shows engineering skills)
  5. Connect to practical project experience ("I implemented this in MiniMind...")

Congratulations on completing all 27 lessons!
""")
