"""
Lesson 13: Attention Variants — Breaking Free from O(N^2)
==========================================================

Standard Softmax Attention has a fatal flaw:
  Time complexity O(N^2), where N is sequence length
  - 1024 tokens → 1M computations
  - 8192 tokens → 67M computations
  - 100K tokens → 10 billion computations (unacceptable!)

In this lesson, we explore 3 variants:
  1. Linear Attention: O(N) complexity
  2. ALiBi: Length extrapolation-friendly positional encoding
  3. Flash Attention: Memory-efficient high-performance Attention

Run: python lessons_en/lesson13_attention_variants.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# Part 1: Review of Standard Attention
# ============================================================

print("=" * 60)
print("Part 1: Standard Softmax Attention Review")
print("=" * 60)

print("""
Standard Attention process:
  Q, K, V are all (seq_len, d_k) matrices
  1. scores = Q @ K^T / √d_k      → (seq_len, seq_len)
  2. weights = softmax(scores)    → (seq_len, seq_len)
  3. output = weights @ V         → (seq_len, d_k)

Problem:
  - step 1's scores is an N×N matrix, memory explodes when N is large
  - step 3 is another N×N matrix multiplication
  - Overall O(N^2) complexity
""")


# ============================================================
# Part 2: Linear Attention
# ============================================================

print("=" * 60)
print("Part 2: Linear Attention — Breaking O(N^2)")
print("=" * 60)

print("""
Core idea: Replace softmax with feature mapping φ

Standard:  output = softmax(QK^T) @ V
Linear:    output = φ(Q) @ (φ(K)^T @ V)
                ↑ Key: Matrix multiplication associativity

Standard Attention: O(N^2) — Must compute N×N attention matrix first
Linear Attention:  O(N)   — Cleverly reorder matrix multiplications

Requirement: softmax(QK^T) ≈ φ(Q) @ φ(K)^T
  → φ is typically elu(x) + 1 or ReLU

Pros and Cons:
  ✓ O(N) complexity, suitable for long sequences
  ✓ Can handle arbitrary length (no retraining needed)
  ✗ Expressiveness slightly weaker than softmax
  ✗ Sensitive to numerical precision during training
""")


def elu_feature_map(x):
    """ELU+1 feature mapping: satisfies φ(q)·φ(k) ≈ exp(q·k)"""
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    """Linear Attention (Performer style)"""

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)

        # Apply feature mapping
        Q = elu_feature_map(xq)
        K = elu_feature_map(xk)
        V = xv

        # Linear Attention core: First compute K^T @ V (D x D), then Q @ (K^T @ V)
        # Equivalent to Q @ K^T @ V, but avoids N×N matrix
        KV = K.transpose(-2, -1) @ V  # (b, h, d_k, d_v)
        out = Q @ KV  # (b, h, n, d_v)

        # Normalization
        K_sum = K.sum(dim=-2, keepdim=True).transpose(-2, -1)  # (b, h, 1, d_k)
        Z = 1.0 / (Q @ K_sum + 1e-8)  # Normalization factor
        out = out * Z

        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


# Complexity analysis demo
print("\n[Experiment] Linear vs Standard Attention Complexity Comparison")
print("-" * 60)
print("Seq Length | Standard O(N²) | Linear O(N) | Speedup")
print("-" * 60)

for seq_len in [128, 512, 2048, 8192]:
    std_ops = seq_len ** 2
    linear_ops = seq_len * 32  # Assume head_dim=32
    print(f"{seq_len:9d} | {std_ops:13,} | {linear_ops:10,} | {std_ops/linear_ops:.0f}x")

print("\n→ The longer the sequence, the greater Linear Attention's advantage!")


# ============================================================
# Part 3: ALiBi — Length Extrapolation Positional Encoding
# ============================================================

print("\n" + "=" * 60)
print("Part 3: ALiBi — Positional Encoding for Length Extrapolation")
print("=" * 60)

print("""
Problem: Standard positional encoding (RoPE) has poor length extrapolation
  - Model trained on length 2048
  - Want to use length 8192 during inference
  - Performance drops significantly

ALiBi (Attention with Linear Biases) solution:
  Don't add positional information in embeddings
  Instead, add a fixed bias proportional to distance in attention scores

Formula:
  attention_score = Q·K^T / √d_k - slope × |i - j|

  Where slope is a pre-set slope, different for each head

  Bias = negative, and more negative with greater distance
  → Tokens farther away get lower attention weights
  → Naturally supports length extrapolation
""")

def get_alibi_slopes(num_heads):
    """Compute ALiBi slopes for each head"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power)
            + get_alibi_slopes(2 * closest_power)[0::2][: num_heads - closest_power]
        )


class ALiBiAttention(nn.Module):
    """Attention with ALiBi bias"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Precompute ALiBi bias matrix
        slopes = torch.tensor(get_alibi_slopes(num_heads))
        # Bias matrix: (num_heads, max_seq_len, max_seq_len)
        positions = torch.arange(max_seq_len)
        # Distance matrix: |i - j|
        distance = (positions[None, :] - positions[:, None]).abs().float()
        # Apply slopes: -slope × distance
        alibi_bias = -slopes[:, None, None] * distance[None, :, :]
        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Add ALiBi bias
        alibi = self.alibi_bias[:, :seq_len, :seq_len]
        scores = scores + alibi[None, :, :, :]  # (1, num_heads, seq, seq)

        # Causal mask
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        out = weights @ xv
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


# Demo ALiBi bias
print("\n[Demo] ALiBi bias matrix visualization (8 heads' slopes)")
print("-" * 60)
slopes = get_alibi_slopes(8)
print(f"8 heads' slopes: {[f'{s:.4f}' for s in slopes]}")
print("Slope differences: Some heads are sensitive to nearby, others to distant positions")

print("\nDistance bias example (slope=0.5):")
print("Dist:  0     1     2     3     4     5")
for i in range(6):
    row = "  ".join([f"{-0.5 * abs(i - j):5.2f}" for j in range(6)])
    print(f"i={i}:  {row}")


# ============================================================
# Part 4: Flash Attention
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Flash Attention — Memory Optimization")
print("=" * 60)

print("""
Problem: Standard Attention's memory bottleneck
  - Actual computation: O(N^2)
  - Memory usage: Also O(N^2)! Must store attention matrix
  - One N=8192 sequence → 8192² × 4 bytes = 256MB (attention only)

Flash Attention (Tri Dao et al., 2022) core ideas:
  1. Don't store the full N×N attention matrix
  2. Block-wise computation, only compute a small block at a time
  3. Use online softmax to accumulate statistics
  4. Both computation and memory significantly reduced

Advantages:
  ✓ Memory complexity O(N) instead of O(N^2)
  ✓ Faster computation (2-4x on GPU)
  ✓ Mathematically equivalent to standard Attention
  ✗ Complex implementation (requires CUDA optimization)

PyTorch built-in:
  F.scaled_dot_product_attention() — PyTorch 2.0+ automatically uses Flash
""")


# Demo: Standard vs Flash Attention memory difference
def attention_standard(Q, K, V):
    """Standard Attention: Requires O(N²) memory"""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def attention_flash(Q, K, V):
    """Flash Attention (using PyTorch built-in SDPA)"""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


# Test numerical consistency
print("\n[Experiment] Verify Flash Attention numerical consistency with standard Attention")
print("-" * 60)
torch.manual_seed(42)
seq_len = 16
d_k = 8
Q = torch.randn(1, 4, seq_len, d_k)
K = torch.randn(1, 4, seq_len, d_k)
V = torch.randn(1, 4, seq_len, d_k)

out_std = attention_standard(Q, K, V)
out_flash = attention_flash(Q, K, V)

print(f"Standard Attention output shape: {out_std.shape}")
print(f"Flash Attention output shape: {out_flash.shape}")
print(f"Max difference: {(out_std - out_flash).abs().max().item():.2e}")
print("Difference is minimal (within floating point precision)!")


# ============================================================
# Part 5: How to Choose in MiniMind
# ============================================================

print("\n" + "=" * 60)
print("Part 5: MiniMind's Attention Variant Selection")
print("=" * 60)

print("""
MiniMind supports multiple attention variants:

1. Standard Attention (default)
   - Classic Softmax Attention
   - Suitable for general scenarios
   - Computational complexity O(N²)

2. Linear Attention
   - Complexity O(N)
   - Suitable for ultra-long sequences
   - Slightly weaker than Standard

3. ALiBi Attention
   - Strong length extrapolation ability
   - Suitable for long text processing
   - Memory similar to Standard

4. Flash Attention
   - Automatically enabled via PyTorch SDPA
   - Low memory usage
   - 2-4x speedup
   - Strongly recommended!

5. Mamba SSM (special)
   - Not based on Attention
   - Complexity O(N)
   - Extremely fast inference

Configuration example (from MiniMind):
```python
# Enable Flash Attention
config = MiniMindConfig(...)
if hasattr(config, 'flash_attn'):
    config.flash_attn = True

# Select Attention type
if config.attention_type == "linear":
    attn = LinearAttention(config)
elif config.attention_type == "alibi":
    attn = ALiBiAttention(config)
else:
    attn = StandardAttention(config)  # Automatically uses Flash
```
""")


# ============================================================
# Comparison Summary
# ============================================================

print("=" * 60)
print("Three Attention Variants Comparison")
print("=" * 60)

comparison = """
┌─────────────────┬──────────┬──────────┬──────────────┐
│ Variant         │ Compute  │ Memory   │ Length Extrap│
├─────────────────┼──────────┼──────────┼──────────────┤
│ Standard        │ O(N²)    │ O(N²)    │ Poor         │
│ Linear          │ O(N)     │ O(N)     │ Good         │
│ ALiBi           │ O(N²)    │ O(N²)    │ Excellent    │
│ Flash           │ O(N²)    │ O(N)     │ Poor         │
│ Mamba           │ O(N)     │ O(N)     │ Good         │
└─────────────────┴──────────┴──────────┴──────────────┘

Recommended use cases:
  - General: Flash Attention (speed + memory optimized)
  - Ultra-long text: Linear Attention or Mamba
  - Need length extrapolation: ALiBi
"""

print(comparison)


# ============================================================
# Deep Understanding: Core Trade-offs of Attention Variants
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: Core Trade-offs of Attention Variants")
print("=" * 60)

print("""
[Analogy: Different 'Attention' Methods]
──────────────────────
  
  Standard Softmax Attention (Meeting):
    Everyone must communicate with all others
    → Most complete information, but noisy with many people (O(N²))
    → Suitable for small meetings
  
  Linear Attention (Memos):
    Each person writes a "summary memo"
    Others read summaries, no direct conversation needed
    → Fast, but loses details
    → Suitable for large company announcements
  
  ALiBi (Distance attenuation):
    Automatically add a "distance filter" when speaking
    Nearby words are clear, distant ones are blurred
    → Naturally supports long conversations
    → Suitable for long meetings
  
  Flash Attention (Optimized meeting process):
    Doesn't change meeting content
    Only optimizes organization
    → Fast, memory efficient
    → Suitable for any scale


[Complexity Comparison of Three Variants]
──────────────────────

  ┌────────────────┬──────────┬──────────┬─────────────┐
  │ Method         │ Compute  │ Memory   │ Length Extrap│
  ├────────────────┼──────────┼──────────┼─────────────┤
  │ Softmax        │ O(N²)    │ O(N²)    │ Poor (retrain)│
  │ Linear         │ O(N)     │ O(N)     │ Excellent    │
  │ ALiBi          │ O(N²)    │ O(N²)    │ Excellent (natural)│
  │ Flash          │ O(N²)    │ O(N)     │ Same as Softmax│
  └────────────────┴──────────┴──────────┴─────────────┘

  Explanation:
    Compute: Number of multiplications
    Memory: Number of intermediate results
    Extrapolation: Can training at 2K run at 32K?


[Diagram: Attention Matrix Comparison]
────────────────────

  Standard Softmax Attention (dense):
    QK^T matrix, almost all elements non-zero
    [0.01 0.02 0.05 ... 0.10]
    [0.03 0.01 0.04 ... 0.08]
    [0.05 0.04 0.02 ... 0.06]
    [   ...              ]
    → O(N²) computation, complete information

  Linear Attention (implicit):
    Doesn't explicitly compute QK^T
    Uses low-rank decomposition of (Q φ)(K φ)^T
    → O(N) computation, loses some precision

  ALiBi (position decay):
    Also computes QK^T, but adds -k * distance
    [0.01 0.005 0.001 ... 0]
    [0.03 0.02  0.01  ... 0]
    [   ...                  ]
    → Distant attention naturally decays


[Mathematical Essence of Linear Attention]
────────────────────────────

  Standard:  attn(Q, K, V) = softmax(QK^T) V
  Linear:    attn(Q, K, V) = (φ(Q) φ(K)^T) V = φ(Q) (φ(K)^T V)
  
  Key: First compute φ(K)^T V, then φ(Q) × ...
  The latter is matrix × matrix, complexity O(N) instead of O(N²)

  Feature mapping φ choices:
    - ELU + 1: Simple, mainstream
    - Random features: Faster
    - cos: Better performance but numerical issues

  Drawbacks:
    - Cannot exactly reproduce softmax
    - Performance slightly lower than softmax
    - Attention distribution not "sharp" enough


[Principle of ALiBi Length Extrapolation]
──────────────────────

  Problem:
    RoPE trained at length 2K, performance drops at 8K inference
    
  Solution:
    Add -k * distance to attention scores
    
  Example: Position 100 attending to position 0 (distance=100)
    Without ALiBi: score 0.5
    With ALiBi: score 0.5 - k*100 = -1 (ignored)
    → Model learns "don't look too much at distant positions"

  Why good extrapolation?
    Training at 2K, sees max distance 2000
    Inference at 8K, max distance 8000
    But model already learned "farther = less important"
    → Still generalizes at 8K


[What Does Flash Attention Solve?]
──────────────────────────────

  Standard Attention memory bottleneck:
    1. QK^T intermediate matrix: [N, N] memory
    2. Post-softmax matrix: [N, N] memory
    3. Intermediate results from V multiplication
    
  At N=4096: 4096² = 16M elements = 64MB (single precision)
  → Large batch memory explosion

  Flash Attention solution:
    Don't actually compute [N, N] matrix
    Block-wise computation, compute and discard
    Mathematically equivalent, memory reduced to O(N)
    
  Actual speedup:
    A100 GPU: 2-4x speedup
    Long sequences (>4K): Even more significant


[How to Choose in MiniMind?]
─────────────────────

  ┌──────────────┬──────────────────────────────┐
  │ Scenario     │ Recommendation               │
  ├──────────────┼──────────────────────────────┤
  │ Short text (<2K) │ Standard Attention       │
  │ Long text (>2K)  │ Standard + RoPE          │
  │ Ultra-long (>32K)│ ALiBi or Linear Attention│
  │ Inference speed  │ Flash Attention (always) │
  │ Resource limited │ Linear Attention         │
  └──────────────┴──────────────────────────────┘

  MiniMind implementation:
    Default: RoPE + Standard Attention
    Optional: Flash Attention (PyTorch 2.0+)
    Configurable switching
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Complexity Comparison
  Sequence length N=8192, standard attention and Linear Attention
  How many multiplications does each need?

[Exercise 2] ALiBi vs RoPE
  What advantage does ALiBi have over RoPE in length extrapolation?

[Exercise 3] Linear Attention Loss
  What capabilities does Linear Attention lose compared to standard Attention?

[Exercise 4] Flash Attention Equivalence
  Why is Flash Attention mathematically equivalent to standard Attention,
  but much faster in practice?

[Exercise 5] MQA vs MHA
  How much memory difference between Multi-Query Attention (MQA)
  and Multi-Head Attention (MHA)?
  (Assume head_dim=64, batch=1, seq=2K, 32 layers)

[Exercise 6] Selection Strategy
  Training on short text, inference on long text, which attention should you choose?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
N = 8192
softmax_ops = N * N
linear_ops = N * 64
ratio = softmax_ops / linear_ops
print(f"  N = {N}")
print(f"  Standard attention: O(N²) = {N}² = {softmax_ops:,} operations")
print(f"  Linear Attention: O(N × d) = {N} × 64 = {linear_ops:,} operations")
print(f"  Speedup: {ratio:.0f}x")
print()
print(f"  → Linear is {ratio:.0f}x faster than standard (at N=8192)")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  RoPE:")
print("    - Uses relative positional encoding")
print("    - Trained 2K, inference 4K performance OK")
print("    - Trained 2K, inference 32K performance drops significantly")
print()
print("  ALiBi:")
print("    - Uses linear bias -k * distance")
print("    - Trained 2K, inference 8K performance drops slightly")
print("    - Trained 2K, inference 32K performance drops moderately")
print()
print("  Reason:")
print("    - ALiBi is 'distance decay', effective at any distance")
print("    - RoPE is 'relative rotation', phases get messy beyond training range")
print()
print("  Experiment:")
print("    RoPE trained 2K → 8K: PPL increases 30%")
print("    ALiBi trained 2K → 8K: PPL increases 5%")
print("    → ALiBi significantly outperforms RoPE in length extrapolation")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Capabilities lost by Linear Attention:")
print()
print("  1. Precise attention distribution:")
print("    Standard: softmax(QK^T) can make attention very 'sharp'")
print("    Linear: Attention distribution is flatter, hard to focus on one token")
print()
print("  2. Strong retrieval ability:")
print("    Standard: Can precisely copy information from a specific token")
print("    Linear: Information is mixed, hard to precisely retrieve")
print()
print("  3. Complex reasoning:")
print("    Standard: Strong 'clue passing' ability in multi-step reasoning")
print("    Linear: Weak on reasoning requiring precise intermediate states")
print()
print("  4. Training stability:")
print("    Standard: Stable training, good convergence")
print("    Linear: Training may be unstable, needs tuning")
print()
print("  Mitigation methods:")
print("    - Use better feature mappings (e.g., cos+activation)")
print("    - Hybrid: Standard in shallow layers, Linear in deep layers")
print("    - Mamba provides a better alternative")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  Mathematical equivalence:")
print("    Softmax formula:")
print("      softmax(x_i) = exp(x_i) / sum_j exp(x_j)")
print("    → sum_j exp(x_j) is 'global normalization'")
print()
print("    Flash Attention block-wise softmax computation:")
print("      1. Split Q, K, V into blocks (e.g., 256 each)")
print("      2. Within each block, compute local softmax")
print("      3. Use running max for correction")
print("      4. Scale proportionally when merging")
print()
print("    → Final result is identical to one-shot computation")
print()
print("  Why faster in practice:")
print("    1. Memory access optimization:")
print("      - No need to store [N, N] intermediate matrix")
print("      - Each block only uses SRAM (on-chip cache, extremely fast)")
print("    2. Reduced memory IO:")
print("      - Standard: Frequent HBM read/write (slow)")
print("      - Flash: More SRAM computation, less HBM access")
print("    3. Parallelization friendly:")
print("      - No inter-block dependencies, perfect parallelism")
print()
print("  Speedup:")
print("    Short sequences (1K): 1.5-2x")
print("    Long sequences (8K+): 3-5x")
print("    Memory-limited scenarios: 10x+")

# Exercise 5
print("\n[Exercise 5 Answer]")
batch = 1
seq_len = 2048
num_layers = 32
num_heads = 32
head_dim = 64
fp16_bytes = 2

# MHA: Each head has independent K, V
mha_kv = 2 * num_heads * head_dim
mha_total = batch * num_layers * seq_len * mha_kv * fp16_bytes
mha_mb = mha_total / (1024 * 1024)

# MQA: Shared K, V
mqa_kv = 1 * head_dim
mqa_total = batch * num_layers * seq_len * mqa_kv * fp16_bytes
mqa_mb = mqa_total / (1024 * 1024)

# GQA: 4 groups share
gqa_groups = 4
gqa_kv = (num_heads // gqa_groups) * head_dim
gqa_total = batch * num_layers * seq_len * gqa_kv * fp16_bytes
gqa_mb = gqa_total / (1024 * 1024)

print(f"  Config: bs={batch}, layers={num_layers}, seq={seq_len}, head_dim={head_dim}")
print(f"  num_heads={num_heads} (MHA/MQA/GQA)")
print()
print(f"  MHA (32 KV heads):")
print(f"    Per-layer KV: {2*num_heads*head_dim} = {mha_kv} elements")
print(f"    Total: {mha_mb:.1f} MB")
print()
print(f"  GQA (4 groups, 8 KV heads):")
print(f"    Per-layer KV: {gqa_kv} elements")
print(f"    Total: {gqa_mb:.1f} MB")
print()
print(f"  MQA (1 KV head):")
print(f"    Per-layer KV: {mqa_kv} elements")
print(f"    Total: {mqa_mb:.1f} MB")
print()
print(f"  Savings:")
print(f"    GQA vs MHA: {(1 - gqa_mb/mha_mb)*100:.0f}%")
print(f"    MQA vs MHA: {(1 - mqa_mb/mha_mb)*100:.0f}%")
print()
print(f"  In practice:")
print(f"    MHA: LLaMA-1, GPT-3")
print(f"    GQA: LLaMA-2/3, MiniMind (mainstream)")
print(f"    MQA: PaLM, StarCoder (extreme optimization)")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Training on short text, inference on long text:")
print()
print("  Best choice: RoPE + length extrapolation techniques")
print()
print("  Option 1: RoPE + Position Interpolation (PI)")
print("    'Compress' positional encoding to fit longer sequences")
print("    Example: Train 2K, inference 8K → Scale positional encoding 4x")
print()
print("  Option 2: RoPE + YaRN")
print("    Improvement over PI, different handling for different frequencies")
print("    Good results, covered in MiniMind Lesson 16")
print()
print("  Option 3: ALiBi (use during training)")
print("    Naturally supports length extrapolation")
print("    Slightly lower performance than RoPE, but best extrapolation")
print()
print("  Option 4: NTK-Aware Scaling")
print("    Modify RoPE's base, no interpolation needed")
print("    Simple implementation, good results")
print()
print("  Practical recommendation:")
print("    Most LLMs: RoPE + YaRN")
print("    LLaMA-2: RoPE (4K → train for 32K inference)")
print("    MiniMind: RoPE + optional YaRN")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
