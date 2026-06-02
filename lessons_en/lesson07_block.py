"""
Lesson 7: Transformer Block — Assembling Attention + FFN
=========================================================

Previous lessons covered individual components:
  - RMSNorm: normalization, stabilizes values
  - RoPE: rotary position encoding, lets the model know positions
  - Attention: words exchange information with each other
  - FFN: each position is processed independently

Now let's assemble them into a complete Transformer Block!

Structure (Pre-Norm architecture, used by MiniMind):

  Input x
    |
    +-> RMSNorm -> Attention -> -> + (Residual Connection)
    |                               |
    |                               +-> RMSNorm -> FFN -> -> + (Residual Connection)
    |                               |                        |
    +-------------------------------+------------------------+
                                                             |
                                                           Output

Key design:
  1. Pre-Norm: normalize before computation (more stable than Post-Norm)
  2. Residual Connection: input is directly added to output, preventing information loss
  3. Two RMSNorms: one before Attention, one before FFN

Run: python lessons_en/lesson07_block.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Reuse components from previous lessons
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def precompute_freqs_cis(dim, end=2048, rope_base=1e6):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    cos_h, sin_h = cos[..., :half], sin[..., :half]
    q_out = torch.cat([q1 * cos_h - q2 * sin_h, q2 * cos_h + q1 * sin_h], dim=-1)
    k_out = torch.cat([k1 * cos_h - k2 * sin_h, k2 * cos_h + k1 * sin_h], dim=-1)
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, seq_len, self.num_heads, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores += mask
        weights = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq))
        output = weights @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Step 1: Assemble the Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """A complete Transformer Block (Pre-Norm architecture)"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        # Step 1: Attention sub-layer
        residual = x
        x_norm = self.input_layernorm(x)
        attn_out = self.self_attn(x_norm, cos, sin)
        x = residual + attn_out  # Residual connection

        # Step 2: FFN sub-layer
        residual = x
        x_norm = self.post_attention_layernorm(x)
        ffn_out = self.mlp(x_norm)
        x = residual + ffn_out   # Residual connection

        return x


# Experiment 1: Block Forward Pass
print("=" * 60)
print("Experiment 1: Transformer Block Forward Pass")
print("=" * 60)

hidden_size = 128
num_heads = 4
num_kv_heads = 2
head_dim = 32
intermediate_size = 256
seq_len = 16
batch_size = 2

block = TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)

freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=2048)
cos = freqs_cos[:seq_len]
sin = freqs_sin[:seq_len]

x = torch.randn(batch_size, seq_len, hidden_size)
out = block(x, cos, sin)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Block doesn't change the shape, only the content!")

param_count = sum(p.numel() for p in block.parameters())
print(f"Block parameter count: {param_count:,}")


# ============================================================
# Step 2: The Role of Residual Connections
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: Residual Connections — Why Are They Essential?")
print("=" * 60)

class BlockNoResidual(nn.Module):
    """Block without residual connections"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        x = self.self_attn(self.input_layernorm(x), cos, sin)
        x = self.mlp(self.post_attention_layernorm(x))
        return x


block_with_res = TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
block_no_res = BlockNoResidual(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)

x = torch.randn(1, 8, hidden_size)

# Stack multiple Blocks, observe signal propagation
x_res = x.clone()
x_no_res = x.clone()

print("Signal strength after stacking 5 Blocks:")
print(f"{'Layer':>4} | {'With residual std':>16} | {'Without residual std':>20} | {'With residual max':>16} | {'Without residual max':>20}")
print("-" * 80)

for i in range(5):
    x_res = block_with_res(x_res, cos[:8], sin[:8])
    x_no_res = block_no_res(x_no_res, cos[:8], sin[:8])
    print(f"  {i+1:>2} | {x_res.std():>16.4f} | {x_no_res.std():>20.4f} | {x_res.abs().max():>16.4f} | {x_no_res.abs().max():>20.4f}")

print("\n-> With residual: signal is stable, won't vanish or explode")
print("-> Without residual: signal may rapidly decay or explode")


# ============================================================
# Step 3: Pre-Norm vs Post-Norm
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: Pre-Norm vs Post-Norm")
print("=" * 60)

class PostNormBlock(nn.Module):
    """Post-Norm architecture: compute first, then normalize"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.input_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x, cos, sin):
        x = self.input_layernorm(x + self.self_attn(x, cos, sin))
        x = self.post_attention_layernorm(x + self.mlp(x))
        return x


print("Pre-Norm (used by MiniMind):")
print("  x = x + Attention(Norm(x))")
print("  x = x + FFN(Norm(x))")
print("  Normalization before computation -> more stable gradients, easier training")

print("\nPost-Norm (original Transformer):")
print("  x = Norm(x + Attention(x))")
print("  x = Norm(x + FFN(x))")
print("  Normalization after computation -> requires careful tuning, harder to train")

print("\nPre-Norm advantages:")
print("  1. Gradients can propagate directly through the residual path (bypassing Norm)")
print("  2. More stable training, no warmup needed")
print("  3. Deep networks can still converge")


# ============================================================
# Step 4: Step-by-Step Tracking of Block's Internal Data Flow
# ============================================================

print("\n" + "=" * 60)
print("Experiment 4: Step-by-Step Tracking of Block's Internal Data Flow")
print("=" * 60)

block = TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
x = torch.randn(1, 4, hidden_size)

print(f"Input x: shape={x.shape}, std={x.std():.4f}")

# Step 1: RMSNorm
residual = x
x_norm = block.input_layernorm(x)
print(f"1. RMSNorm: std={x_norm.std():.4f}")

# Step 2: Attention
attn_out = block.self_attn(x_norm, cos[:4], sin[:4])
print(f"2. Attention output: std={attn_out.std():.4f}")

# Step 3: Residual connection
x_after_attn = residual + attn_out
print(f"3. After residual connection: std={x_after_attn.std():.4f}")

# Step 4: RMSNorm
residual2 = x_after_attn
x_norm2 = block.post_attention_layernorm(x_after_attn)
print(f"4. RMSNorm: std={x_norm2.std():.4f}")

# Step 5: FFN
ffn_out = block.mlp(x_norm2)
print(f"5. FFN output: std={ffn_out.std():.4f}")

# Step 6: Residual connection
x_after_ffn = residual2 + ffn_out
print(f"6. After residual connection: std={x_after_ffn.std():.4f}")

print(f"\nComplete Block output: shape={x_after_ffn.shape}, std={x_after_ffn.std():.4f}")


# ============================================================
# Step 5: Block Parameter Count Analysis
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5: Block Parameter Distribution")
print("=" * 60)

hidden_size = 768
num_heads = 12
num_kv_heads = 4
head_dim = 64
intermediate_size = 2048

block = TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)

attn_params = sum(p.numel() for p in block.self_attn.parameters())
ffn_params = sum(p.numel() for p in block.mlp.parameters())
norm_params = sum(p.numel() for m in [block.input_layernorm, block.post_attention_layernorm] for p in m.parameters())
total_params = attn_params + ffn_params + norm_params

print(f"Attention parameters: {attn_params:>10,} ({attn_params/total_params*100:.1f}%)")
print(f"FFN parameters:       {ffn_params:>10,} ({ffn_params/total_params*100:.1f}%)")
print(f"RMSNorm parameters:   {norm_params:>10,} ({norm_params/total_params*100:.1f}%)")
print(f"Total:                {total_params:>10,}")
print(f"\n-> FFN accounts for the majority of parameters!")


# ============================================================
# Step 6: Stacking Multiple Blocks
# ============================================================

print("\n" + "=" * 60)
print("Experiment 6: Stacking Multiple Blocks")
print("=" * 60)

num_layers = 8
blocks = nn.ModuleList([
    TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
    for _ in range(num_layers)
])

x = torch.randn(1, 16, hidden_size)
cos = freqs_cos[:16]
sin = freqs_sin[:16]

print(f"Input: shape={x.shape}, std={x.std():.4f}")

for i, block in enumerate(blocks):
    x = block(x, cos, sin)
    if i % 2 == 0:
        print(f"Block {i+1} output: std={x.std():.4f}, max={x.abs().max():.4f}")

print(f"\nAfter {num_layers} Blocks: shape={x.shape}")
print("Signal remains stable, thanks to Pre-Norm + Residual Connections!")

total_model_params = sum(p.numel() for p in blocks.parameters())
print(f"{num_layers} Blocks total parameters: {total_model_params:,} = {total_model_params/1e6:.2f}M")


# ============================================================
# Deep Dive: The Essence of Transformer Block
# ============================================================
print("\n" + "=" * 60)
print("Deep Dive: The Essence of Transformer Block")
print("=" * 60)

print("""
[Block's Two Core Functions]
----------------------------
Think of a Block as an "information processing unit":

  +---------------------------------------------------+
  |  Block functions:                                  |
  |    1. Attention: cross-position information exchange|
  |    2. FFN:        single-position processing       |
  +---------------------------------------------------+

  Attention lets words exchange information (horizontal)
  FFN lets each word think independently (vertical)
  Residual connections preserve original signals, avoiding information loss


[Intuitive Explanation of Residual Connections]
-----------------------------------------------
Without residual:  y = F(x)
  -> After x passes through F, original information may be lost
  -> After multiple layers, signal is completely distorted

With residual:  y = x + F(x)
  -> No matter what F learns, x is always preserved
  -> Deep networks can also preserve original signals
  -> Analogy: when making copies, the "original" is always kept,
     while "copies" are continuously processed

Mathematically:
  dL/dx = dL/dy * (1 + dF/dx)
  -> Gradient always has the direct propagation channel of 1
  -> No vanishing gradients!


[Diagram: Data Flow vs Gradient Flow]
--------------------------------------

Data Flow (Forward Pass):

  x ------------------------------------+
  |                                     | <- residual
  v                                     |
RMSNorm                                 |
  |                                     |
  v                                     |
Attention                               |
  |                                     |
  +--(+)--------------------------------+
  |                                     |
RMSNorm                                 |
  |                                     |
  v                                     |
FFN                                     |
  |                                     |
  +--(+)--------------------------------+
  |
  v
  y

Gradient Flow (Backpropagation):

  dL/dy
   |
   +--(+)-----------------------> dL/dx (direct propagation)
   |
   v
  dL/dFFN(...)
   |
   +--(+)-----------------------> dL/dx
   |
   v
  dL/dRMSNorm(...)
  ...


[Essential Difference: Pre-LN vs Post-LN]
------------------------------------------
Pre-LN (used by MiniMind):
  x -> LN -> Attn -> + -> LN -> FFN -> +
       normalize  process  ^  normalize  process  ^
                 residual             residual
  Residual path: x -> x (identity mapping, clean)
  Sub-layer input: always normalized by LN (stable range)

Post-LN (original Transformer):
  x -> Attn -> + -> LN -> FFN -> + -> LN
       process  ^  normalize  process  ^  normalize
              residual            residual
  Residual path: x -> + Attn -> + FFN -> ... (accumulated)
  Sub-layer input: not normalized, range explodes in deep layers

Experimental comparison:
  Pre-LN:  stable training, 100+ layers can be trained
  Post-LN: difficult training, 12 layers may have exploding gradients


[Division of Labor Between Blocks]
-----------------------------------
Shallow Blocks (1-3):
  - Learn local grammar (parts of speech, dependencies)
  - Small attention range
  - Low feature abstraction level

Middle Blocks (4-7):
  - Learn syntactic structure (subject-verb-object, modifiers)
  - Medium attention range
  - Features begin to abstract

Deep Blocks (8+):
  - Learn semantics and reasoning
  - Large attention range
  - Highly abstract features

Analogy: the cerebral cortex is also layered
  - Primary visual cortex: edges, colors
  - Secondary visual cortex: shapes, textures
  - Higher visual cortex: objects, scenes
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Role of Residual Connections
  What happens without residual connections?
  What if we use a 1x1 convolution (small parameters) as the residual?

[Exercise 2] Pre-LN vs Post-LN
  Why is Post-LN hard to train? How to understand this mathematically?

[Exercise 3] Number of Blocks
  MiniMind-26M has 8 Block layers, GPT-3 has 96
  Why do larger models need more Blocks?

[Exercise 4] Parameter Allocation Within a Block
  Given hidden=512, inter=2048, num_heads=8
  How many parameters do Attention and FFN each occupy in a single Block?

[Exercise 5] Signal Propagation
  50 layers of Pre-LN + residual, input x, output y
  How many layers of x's "original information" does y contain?

[Exercise 6] Block Reusability
  What problems arise if the same Block is reused 12 times?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Without residual:")
print("    - After 50 layers, signal is almost completely vanished or exploded")
print("    - Vanishing gradients, training fails")
print()
print("  1x1 convolution residual (i.e., y = W*x + F(x), W is not identity):")
print("    - Better than no residual, but still has gradient issues")
print("    - W is trained, may become very small (vanishing) or very large (exploding)")
print("    - Loses the elegant property of 'identity mapping'")
print()
print("  Identity residual (y = x + F(x)) is the simplest and most stable")
print("  -> The core of residual is 'identity mapping', any parameterization loses this property")

# Demo
import torch
import torch.nn as nn

class NoResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class IdentityResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))

x = torch.randn(1, 1, 64)
m_no = NoResidual(64)
m_id = IdentityResidual(64)
print(f"  No residual output norm: {m_no(x).norm():.4f}")
print(f"  Identity residual output norm: {m_id(x).norm():.4f}")
print(f"  Input norm: {x.norm():.4f}")
print(f"  -> Identity residual output is on the same scale as input, no residual may differ greatly")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Why Post-LN is hard to train:")
print()
print("  Mathematical analysis:")
print("    Post-LN output: y = LN(F(x) + x)")
print("    Variance along residual path accumulates with depth: Var(x_L) = L * Var(x_0)")
print("    -> When L=50, variance is amplified 50x")
print("    -> Sub-layers need to handle increasingly large inputs")
print("    -> Weights need constant range adjustment, training is unstable")
print()
print("  Pre-LN advantage:")
print("    y = x + F(LN(x))")
print("    Residual path variance is constant: Var(x_L) = Var(x_0)")
print("    Sub-layer input is always a normalized, stable distribution")
print("    -> Stable training, 100+ layers can be trained")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Relationship between Block count and model capability:")
print()
print("  Analogy: thinking about a problem")
print("    Shallow: first reaction (intuitive, but rough)")
print("    Middle: careful thinking (analysis)")
print("    Deep: repeated deliberation (reasoning, reflection)")
print()
print("  Large models need more Blocks:")
print("    - Handle more complex tasks (reasoning, planning)")
print("    - Memorize more knowledge")
print("    - In-context learning")
print()
print("  Reference counts:")
print("    GPT-2:   12-48 layers")
print("    GPT-3:   96 layers (175B parameters)")
print("    LLaMA-7B:  32 layers")
print("    LLaMA-65B: 80 layers")
print("    MiniMind:  8 layers (toy-level)")

# Exercise 4
print("\n[Exercise 4 Answer]")
hidden = 512
inter = 2048
num_heads = 8

attn = 4 * hidden * hidden  # Q, K, V, O
print(f"  Attention: 4 x {hidden}^2 = {attn:,} = {attn/1e6:.2f}M")

ffn = 3 * hidden * inter  # gate, up, down (SwiGLU)
print(f"  FFN (SwiGLU): 3 x {hidden} x {inter} = {ffn:,} = {ffn/1e6:.2f}M")

norm = 2 * hidden
print(f"  RMSNorm (2): 2 x {hidden} = {norm:,}")

total = attn + ffn + norm
print(f"  Block total: {total:,} = {total/1e6:.2f}M")
print(f"  Attention proportion: {attn/total*100:.1f}%")
print(f"  FFN proportion: {ffn/total*100:.1f}%")
print(f"  -> FFN accounts for ~{ffn/total*100:.0f}% of Block parameters")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Mathematical analysis (Pre-LN + residual):")
print()
print("    y_L = x_0 + sum_{{i=0}}^{{L-1}} F_i(LN(x_i))")
print()
print("  Each layer contributes one F_i(LN(x_i))")
print("  So y_L contains:")
print("    - 1 copy of x_0 (original signal)")
print("    - L copies of different transformations")
print()
print("  Analogy: cake + toppings")
print("    x_0 is the cake base (foundation)")
print("    F_i is the cream/fruit added at each layer (decoration)")
print("    After 50 layers, the cake base is still there, just richly decorated")
print()
print("  Key: x_0 always exists, this is the benefit of Pre-LN residual")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Problems with reusing the same Block:")
print()
print("  1. Limited expressiveness:")
print("    - Equivalent to stacking 12 identical functions")
print("    - Equivalent to composing one function 12 times")
print("    - Far less expressive than 12 different functions learning diverse features")
print()
print("  2. Training difficulty:")
print("    - Identical gradient paths, prone to symmetry breaking issues")
print("    - In practice, all layers would learn the same thing")
print()
print("  3. Analogy:")
print("    12 people using the same machine vs 12 people each with their own")
print("    The latter is much more efficient")
print()
print("  In practice: must use different Blocks with independent weights")
print("  -> Different Blocks naturally learn different patterns")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
print("""
1. Transformer Block = RMSNorm + Attention + Residual + RMSNorm + FFN + Residual
2. Residual connections allow signals to propagate directly, preventing vanishing gradients in deep networks
3. Pre-Norm is more stable than Post-Norm, MiniMind uses Pre-Norm
4. FFN accounts for the majority of Block parameters
5. Stacking multiple Blocks forms deep networks, signals still propagate stably

Complete data flow:
  Input x
    -> RMSNorm -> Attention -> +x (residual)
    -> RMSNorm -> FFN -> +x (residual)
    -> Output (passed to the next Block)

Next -> lesson08_gpt.py: Assemble multiple Blocks into a complete language model
""")
