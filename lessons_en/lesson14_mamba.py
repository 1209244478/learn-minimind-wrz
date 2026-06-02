"""
Lesson 14: Mamba — The Architecture Challenging Attention
==========================================================

Mamba (Albert Gu & Tri Dao, 2023) is a powerful challenger to Transformer.

Core innovations:
  1. State Space Model (SSM) as foundation
  2. Selective mechanism (Selective SSM) — lets the model "filter" important info
  3. Linear complexity O(N) — beats Attention's O(N²)

This lesson covers:
  1. What is a State Space Model
  2. How Selective Scan works
  3. Complete Mamba Block structure
  4. How to use it in MiniMind

Run: python lessons_en/lesson14_mamba.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: What is a State Space Model (SSM)
# ============================================================

print("=" * 60)
print("Part 1: What is a State Space Model (SSM)")
print("=" * 60)

print("""
State Space Models come from control theory:
  Describing how a system evolves over time

Continuous form:
  h'(t) = A·h(t) + B·x(t)   # State update
  y(t)  = C·h(t) + D·x(t)   # Output

Where:
  x(t) = input
  h(t) = hidden state (compresses all historical info)
  y(t) = output
  A, B, C, D = system parameters

Analogy:
  x(t) = water flowing in
  h(t) = pool
  y(t) = water flowing out
  A = pool's "leak rate"
  B = inlet valve
  C = outlet valve

Why is SSM suitable for sequence modeling?
  - Hidden state h(t) can compress arbitrary-length history → fixed size!
  - High computational efficiency
  - Can be parallelized during training (using convolution form)
  - O(1) per step during inference
""")


# ============================================================
# Part 2: S4 — Efficient SSM Implementation
# ============================================================

print("\n" + "=" * 60)
print("Part 2: S4 — Convolution for Training, Recurrence for Inference")
print("=" * 60)

print("""
S4 (Structured State Space) key insight:
  SSM can be computed in two ways:
    1. Recurrent form (suitable for inference)
       h_t = A·h_{t-1} + B·x_t
       y_t = C·h_t

    2. Convolution form (suitable for training)
       y = x * K    (where K = (C·B, C·A·B, C·A²·B, ...))

  Training: Compute K, use convolution for parallel computation
  Inference: Switch to recurrent form, O(1) per step
""")


def simple_ssm_recurrent(A, B, C, x):
    """Recurrent form SSM - suitable for inference (1D simplified)"""
    seq_len = x.shape[0]
    h = torch.tensor(0.0)
    outputs = []
    for t in range(seq_len):
        h = A * h + B * x[t]
        y = C * h
        outputs.append(y)
    return torch.stack(outputs)


def simple_ssm_conv(A, B, C, x):
    """Convolution form SSM - suitable for training (1D simplified)"""
    seq_len = x.shape[0]
    K = []
    A_pow = torch.tensor(1.0)
    for t in range(seq_len):
        K.append((C * B * A_pow).item())
        A_pow = A_pow * A
    K = torch.tensor(K)
    y = torch.nn.functional.conv1d(
        x.view(1, 1, -1), K.view(1, 1, -1)
    ).view(-1)
    return y


print("\n[Demo] Recurrent vs Convolution SSM numerical consistency")
print("-" * 60)
A = torch.tensor(0.8)
B = torch.tensor(0.3)
C = torch.tensor(0.5)
x = torch.randn(10)

y_recurrent = simple_ssm_recurrent(A, B, C, x)
y_conv = simple_ssm_conv(A, B, C, x)

print(f"Recurrent output: {[f'{v:.3f}' for v in y_recurrent[:5].tolist()]}")
print(f"Convolution output: {[f'{v:.3f}' for v in y_conv[:5].tolist()]}")
print(f"Max difference: {(y_recurrent - y_conv).abs().max():.2e}")


# ============================================================
# Part 3: Mamba's Core Innovation — Selective Mechanism
# ============================================================

print("\n" + "=" * 60)
print("Part 3: Mamba's Core Innovation — Selective Mechanism")
print("=" * 60)

print("""
Problem with traditional SSM:
  A, B, C, D are fixed, independent of input
  → Model cannot "dynamically adjust" based on input
  → Cannot selectively remember or forget

Mamba's solution: Make B, C, Δ depend on input!
  B, C, Δ = Linear(x)   # Dynamically generated from input

  Δ (delta) is the key innovation — it controls "time step"
  - Δ large: Current input is important, fast state update
  - Δ small: Current input is unimportant, preserve history

Formula:
  B_t = Linear_B(x_t)
  C_t = Linear_C(x_t)
  Δ_t = softplus(Linear_Δ(x_t))   # Ensure > 0
  h_t = exp(A · Δ_t) · h_{t-1} + Δ_t · B_t · x_t
  y_t = C_t · h_t

Analogy:
  Traditional SSM = Fixed-size pool, fixed flow rate
  Mamba          = Smart pool, flow rate adjusts with content
                     - Important info → Fast inflow
                     - Unimportant → Slow inflow
""")


# ============================================================
# Part 4: Complete Mamba Block Implementation
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Complete Mamba Block Implementation")
print("=" * 60)


class MambaBlock(nn.Module):
    """Simplified Mamba Block - demonstrating core structure"""

    def __init__(self, hidden_size, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = hidden_size * expand

        self.norm = nn.LayerNorm(hidden_size)

        self.in_proj = nn.Linear(hidden_size, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).expand(self.d_inner, -1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        x_proj = self.x_proj(x)
        B = x_proj[:, :, :self.d_state]
        C = x_proj[:, :, self.d_state:2*self.d_state]
        dt = x_proj[:, :, -1:]

        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log)

        h = torch.zeros(bsz, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(seq_len):
            dA = torch.exp(A * dt[:, t].unsqueeze(-1))
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
            h = dA * h + dB * x[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)

        y = y + self.D * x

        y = y * F.silu(z)

        return residual + self.out_proj(y)


print("\n[Experiment] Mamba Block shape verification")
print("-" * 60)
mamba = MambaBlock(hidden_size=64, d_state=16)
x = torch.randn(2, 16, 64)
out = mamba(x)
print(f"Input: {x.shape}  →  Output: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in mamba.parameters()):,}")


# ============================================================
# Part 5: Mamba vs Attention Comparison
# ============================================================

print("\n" + "=" * 60)
print("Part 5: Mamba vs Attention")
print("=" * 60)

comparison = """
┌──────────────────┬─────────────────┬────────────────┐
│ Dimension        │ Attention       │ Mamba          │
├──────────────────┼─────────────────┼────────────────┤
│ Complexity (train)│ O(N²)          │ O(N log N)     │
│ Complexity (infer)│ O(N²)          │ O(1) per step  │
│ Memory           │ O(N²)           │ O(N)           │
│ Long context     │ Difficult       │ Strong         │
│ Retrieval ability│ Strong          │ Weak           │
│ Inference speed  │ Limited by KV Cache │ Extremely fast│
│ Training parallel│ Fully parallel  │ Selective scan │
└──────────────────┴─────────────────┴────────────────┘

Mamba advantages:
  ✓ O(1) per step during inference (just maintain state)
  ✓ Low memory usage (no KV Cache)
  ✓ Very suitable for long sequences
  ✓ Training speed comparable to Transformer

Mamba disadvantages:
  ✗ Cannot precisely "retrieve" (unlike Attention's query)
  ✗ Complex selective scan implementation
  ✗ Still requires recurrence during training (partially solved via parallel algorithms)
  ✗ Weaker at in-context learning

Hybrid approach (used by MiniMind):
  Lower layers use Mamba (fast processing of long sequences)
  Upper layers use Attention (precise global modeling)
"""

print(comparison)


# ============================================================
# Part 6: How to Use in MiniMind
# ============================================================

print("\n" + "=" * 60)
print("Part 6: Mamba Hybrid Architecture in MiniMind")
print("=" * 60)

print("""
MiniMind supports Mamba + Attention hybrid architecture:

  [Mamba × 3] → [Mamba × 3] → [Attention × 2] → [Attention × 2]
       Fast local processing        Precise global modeling

Configuration example:
```python
config = MiniMindConfig(
    hidden_size=512,
    num_hidden_layers=8,
    mamba_hybrid=True,        # Enable hybrid architecture
    mamba_ratio=0.5,          # First 50% use Mamba
    d_state=16,               # Mamba state dimension
)
```

Code:
```python
class MiniMindBlock:
    def __init__(self, config, layer_id):
        if config.mamba_hybrid and layer_id < int(config.num_hidden_layers * config.mamba_ratio):
            self.self_attn = MambaLayer(config)  # Mamba
        else:
            self.self_attn = Attention(config)    # Standard Attention
```
""")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. State Space Model (SSM)
   - Continuous differential equation: h'=A·h+B·x, y=C·h
   - Hidden state compresses entire history → fixed size
   - Use convolution form for training, recurrent form for inference

2. Mamba's Core Innovation
   - Selective mechanism: B, C, Δ depend on input
   - Lets model "dynamically choose" what to remember and forget
   - Hardware-friendly parallel scan algorithm

3. Mamba Block Structure
   - 1D convolution (local information)
   - Selective SSM (long-range dependencies)
   - Gating mechanism

4. Mamba vs Attention
   - Mamba: O(N) complexity, fast inference, no KV Cache
   - Attention: O(N²) complexity, strong retrieval ability
   - Hybrid architecture: Best of both worlds

Next: Lesson 15 - LoRA Fine-tuning (Efficient Adaptation of Large Models)
""")


# ============================================================
# Deep Understanding: Mamba's "Selective Memory"
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: Mamba's 'Selective Memory'")
print("=" * 60)

print("""
[Analogy: Mamba = Smart Notebook]
──────────────────────

  Transformer (Attention):
    Like a meeting: Every sentence requires reviewing all history
    → Complete info, but slow

  RNN/LSTM:
    Like a brain: Speak and remember simultaneously
    → Fast, but forgets details

  Mamba (S6):
    Like a smart notebook with a filter:
      - Remember current important info
      - Forget unimportant things
      - Quickly flip to any page when needed
    → Combines speed and memory capability

  Specifically:
    h_t = A * h_{t-1} + B * x_t      ← Update memory
    y_t = C * h_t                      ← Read memory

    Where A, B, C are functions of input (key innovation!)


[Core of Mamba: Selective SSM (S6)]
────────────────────────────────

  Traditional SSM (S4):
    A, B, C are fixed parameters
    → All inputs use the same memory strategy
    → Poor adaptability

  Mamba (S6):
    A, B, C are dynamically generated from input x
    → Different inputs use different memory strategies
    → Extremely strong adaptability

  This is why Mamba can "selectively" remember/forget:
    - See keywords → Large B, strong update
    - See filler → Small B, weak update
    - Long-term memory → Appropriate A, slow decay
    - Short-term memory → Fast decay, quick clearing


[Diagram: SSM State Update]
────────────────────

  Time step t=1:    t=2:    t=3:    t=4:

  x_1 → [B_1]    x_2 → [B_2]    x_3 → [B_3]    x_4 → [B_4]
       ↗ ↓ ↘        ↗ ↓ ↘        ↗ ↓ ↘        ↗ ↓ ↘
  [A_1] [h_1]    [A_2] [h_2]    [A_3] [h_3]    [A_4] [h_4]
       ↘ ↓ ↗        ↘ ↓ ↗        ↘ ↓ ↗        ↘ ↓ ↗
       y_1          y_2          y_3          y_4

  Formula:
    h_t = A_t * h_{t-1} + B_t * x_t
    y_t = C_t * h_t

  Where A_t, B_t, C_t are generated from x_t (Mamba innovation)


[Mamba vs Transformer Complexity]
────────────────────────────

  ┌────────────┬──────────┬──────────┬──────────┐
  │ Metric     │ Training │ Inference│ Memory   │
  ├────────────┼──────────┼──────────┼──────────┤
  │ Transformer│ O(N²)    │ O(N) KV  │ O(N²)    │
  │ Mamba      │ O(N)     │ O(1)/step│ O(N)     │
  └────────────┴──────────┴──────────┴──────────┘

  Training:
    Transformer: Each token sees all others (N²)
    Mamba:       Each token processed sequentially (N)
    → Mamba much faster for long sequences

  Inference:
    Transformer: Has KV Cache, O(1) per step
    Mamba:       O(1) per step (no Cache)
    → Comparable speed, but Mamba saves memory

  Memory:
    Transformer: KV Cache takes O(N)
    Mamba:       Only needs current state O(1)
    → Mamba extremely memory-efficient

  In practice:
    Long sequences (16K+): Mamba has huge advantage
    Short sequences (<2K): Little difference


[Mamba's Hardware-Efficient Algorithm]
──────────────────────────────

  Problem: Naive recurrence O(N) cannot be parallelized

  Solution: Parallel Scan
    Turn N serial steps into O(log N) parallel steps

  Example: Accumulating 8 elements
    Serial: 1→2→3→4→5→6→7→8 (8 steps)
    Parallel: (1+2) (3+4) (5+6) (7+8)
             ((1+2)+(3+4)) ((5+6)+(7+8))
             (((1+2)+(3+4))+((5+6)+(7+8)))
             (3 steps, 2.6x faster)

  Actual speedup:
    Sequence 8K:  ~16x speedup
    Sequence 16K: ~25x speedup
    Sequence 32K: ~40x speedup


[Mamba Block vs Transformer Block]
────────────────────────────

  Transformer Block:
    x → RMSNorm → Attention → Add
        → RMSNorm → FFN       → Add
    → output

  Mamba Block:
    x → Norm → in_proj → Conv1d → SSM → out_proj
                ↓
                SiLU (gating)
    → output + residual

  Key differences:
    - No Attention, uses SSM instead
    - Introduces Conv1d for local patterns
    - Uses SiLU gating
    - Overall more lightweight


[Hybrid Architecture: Mamba + Attention]
─────────────────────────────

  Where pure Mamba is weak:
    - Precise retrieval (e.g., "copy a specific token")
    - In-context learning

  Where pure Transformer is weak:
    - Long sequence efficiency
    - Inference memory

  Hybrid (Jamba, Zamba):
    - Some layers use Mamba
    - Some layers use Attention
    - Best of both worlds

  Ratio (empirical):
    6-8 Mamba layers + 1 Attention layer
    → Combines efficiency and precision

  MiniMind:
    Supports configurable mamba_layers / attn_layers ratio
    Default 1:1 hybrid
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] SSM State Dimension
  What is the dimension of Mamba's state h? What does it represent?

[Exercise 2] Roles of A, B, C
  What do A, B, C represent in SSM? Compare with RNN gating mechanisms.

[Exercise 3] Mamba vs LSTM
  What is Mamba's core improvement over LSTM?

[Exercise 4] Mamba Has No KV Cache
  Why doesn't Mamba need KV Cache? What are the benefits?

[Exercise 5] Mamba Long Sequence Advantage
  Sequence N=32768, how much memory difference between Mamba and Transformer?
  (Mamba state [B, D, N] vs Transformer KV [B, L, N, H, D])

[Exercise 6] Hybrid Architecture
  Why can't we use all Mamba? Why do we need Attention?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Mamba state h dimension: [batch, d_inner, d_state]")
print("    d_inner:  Inner dimension (similar to hidden_size)")
print("    d_state:  State dimension (default 16, relatively small)")
print()
print("  Meaning:")
print("    - d_state = 16 seems small, but is powerful")
print("    - Each dimension can be thought of as 'a memory unit'")
print("    - 16 memory units can represent complex patterns")
print()
print("  Comparison:")
print("    LSTM state: Hundreds to thousands of dimensions")
print("    Mamba state: 16 dimensions (more compact)")
print()
print("  Why is 16 enough?")
print("    - A, B, C are functions of input, self-adaptive")
print("    - No need to store all history, just 'summary'")
print("    - Similar to compressed storage")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  A (state transition matrix):")
print("    - Determines how much old memory to retain")
print("    - A close to 1: Long-term memory")
print("    - A close to 0: Fast forgetting")
print("    - Analogous to LSTM's forget gate")
print()
print("  B (input mapping):")
print("    - Determines weight of new input")
print("    - B large: Current input is important")
print("    - B small: Current input is ignored")
print("    - Analogous to LSTM's input gate")
print()
print("  C (output mapping):")
print("    - Determines how to read from state")
print("    - Similar to LSTM's output gate")
print()
print("  Mamba's innovation:")
print("    A, B, C are all dynamically generated from input")
print("    → More flexible than LSTM's fixed gating")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Mamba's core improvements over LSTM:")
print()
print("  1. State space vs gated recurrence:")
print("    LSTM: Three gates, complex but hard to parallelize")
print("    Mamba: SSM form, mathematically elegant")
print()
print("  2. Selective mechanism:")
print("    LSTM: Gates are 'hard switches' (0/1)")
print("    Mamba: A, B, C are continuous values, smoother")
print()
print("  3. Parallel training:")
print("    LSTM: Must be serial, slow training")
print("    Mamba: Uses Parallel Scan, can be parallelized")
print()
print("  4. Long-range dependencies:")
print("    LSTM: Forgets after a few steps")
print("    Mamba: State mechanism naturally preserves long-range")
print()
print("  5. Hardware efficiency:")
print("    LSTM: Hard to run efficiently on GPU")
print("    Mamba: Designed for GPU (like Attention)")
print()
print("  Performance:")
print("    Same parameters, Mamba much stronger than LSTM")
print("    Training speed: Mamba 5-10x faster")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  Why Mamba doesn't need KV Cache:")
print()
print("  Transformer:")
print("    Attention needs all historical K, V")
print("    Must store them during generation → KV Cache")
print()
print("  Mamba:")
print("    State h_t already 'compresses' all history")
print("    No need to store K, V separately")
print("    Only need current h_t (fixed size)")
print()
print("  Benefits:")
print("    1. Memory savings:")
print("      Transformer: KV Cache takes O(N)")
print("      Mamba:       State takes O(1) (independent of sequence length)")
print()
print("    2. Fast inference:")
print("      No need to concatenate new K, V")
print("      Only need to update h_t each time")
print()
print("    3. Long sequence friendly:")
print("      Transformer: 32K sequence needs 32K steps of Cache")
print("      Mamba:       Always only needs current state")
print()
print("  Actual data:")
print("    32K sequence inference:")
print("      Transformer: Tens of GB Cache")
print("      Mamba:       < 1 MB")

# Exercise 5
print("\n[Exercise 5 Answer]")
N = 32768
batch = 1
d_inner = 1024
d_state = 16
num_layers = 32
num_kv_heads = 8
head_dim = 64
fp16 = 2

mamba_bytes = batch * d_inner * d_state * fp16 * num_layers
mamba_mb = mamba_bytes / (1024 * 1024)

tf_bytes = 2 * batch * num_layers * N * num_kv_heads * head_dim * fp16
tf_mb = tf_bytes / (1024 * 1024)
tf_gb = tf_mb / 1024

print(f"  Config: bs={batch}, N={N}, layers={num_layers}")
print(f"  Mamba state: [{batch}, {d_inner}, {d_state}] = {mamba_bytes/1024:.1f} KB")
print(f"  Per-layer state: {batch * d_inner * d_state * fp16} bytes = {batch * d_inner * d_state * fp16/1024:.1f} KB")
print(f"  Mamba 32-layer total state: {mamba_mb:.1f} MB")
print()
print(f"  Transformer KV:")
print(f"  Per-layer KV: {2 * batch * N * num_kv_heads * head_dim * fp16/1024/1024:.1f} MB")
print(f"  32-layer total KV: {tf_mb:.1f} MB = {tf_gb:.2f} GB")
print()
print(f"  → At 32K sequence, Mamba saves {(1 - mamba_mb/tf_mb)*100:.0f}% memory vs Transformer")
print(f"  → This is the fundamental reason Mamba suits ultra-long sequences")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Why we can't use all Mamba:")
print()
print("  1. Weak precise retrieval:")
print("    Transformer's Attention can precisely 'see' a specific position")
print("    Mamba uses compressed summary, hard to pinpoint")
print("    → Not suitable for: Code completion, reference lookup")
print()
print("  2. Weak complex reasoning:")
print("    Multi-step reasoning requires 'clue passing'")
print("    Mamba's state is compressed, poor passing ability")
print("    → Not suitable for: Math reasoning, logic analysis")
print()
print("  3. With limited training data:")
print("    Mamba needs more data to learn 'selectivity'")
print("    Small data training, Transformer is more stable")
print()
print("  Scenarios needing Attention:")
print("    - Retrieval-augmented generation (RAG)")
print("    - Precise copying/quoting")
print("    - In-context learning")
print()
print("  Hybrid strategy:")
print("    - Even layers: Mamba (efficiency)")
print("    - Odd layers: Attention (precision)")
print("    - Ratio 4:1 or 6:1")
print()
print("  Actual models:")
print("    Jamba: Mamba + Attention 1:7")
print("    Zamba: Alternating use")
print("    MiniMind: Configurable")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
