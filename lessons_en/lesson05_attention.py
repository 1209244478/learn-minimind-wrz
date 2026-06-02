"""
Lesson 5: Attention — Core Mechanism: How Do Words Attend to Each Other?
=========================================================================

Attention is the soul of the Transformer.
In one sentence: each word "queries" all other words and collects information based on relevance.

Core Formula:
  Attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V

Breakdown:
  Q (Query):  "What am I looking for?" — each word's query vector
  K (Key):    "What do I have?"        — each word's key vector
  V (Value):  "My content is"          — each word's value vector

  1. Q·K^T:    Compute relevance score for each word pair
  2. /sqrt(d): Scale to prevent scores from being too large
  3. softmax:  Normalize to probability distribution (attention weights)
  4. ·V:       Weighted sum to get each word's new representation

MiniMind uses GQA (Grouped Query Attention):
  Multiple Q heads share one set of K/V heads, reducing KV cache memory overhead.

Run: python lessons_en/lesson05_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Step 1: Intuitive Understanding of Attention
# ============================================================

print("=" * 60)
print("Experiment 1: Attention Intuition — Information Retrieval")
print("=" * 60)

# Imagine you're in a library looking for books:
# Q = your need: "I want books about cats"
# K = each book's label: "cats", "dogs", "cooking"
# V = each book's content

# Simulate with vectors
query = torch.tensor([1.0, 0.0])      # Need: focus on "cat" dimension
key_cat = torch.tensor([0.9, 0.1])     # Label: mainly cats
key_dog = torch.tensor([0.1, 0.9])     # Label: mainly dogs
key_cook = torch.tensor([0.0, 0.0])    # Label: neither relevant

# Compute relevance scores (dot product)
score_cat = torch.dot(query, key_cat)
score_dog = torch.dot(query, key_dog)
score_cook = torch.dot(query, key_cook)

print(f"Query: {query.tolist()}")
print(f"Relevance with 'cat': {score_cat:.2f}")
print(f"Relevance with 'dog': {score_dog:.2f}")
print(f"Relevance with 'cooking': {score_cook:.2f}")
print("-> Higher relevance -> higher attention weight")


# ============================================================
# Step 2: Manual Single-Head Attention
# ============================================================

def single_head_attention(Q, K, V, mask=None):
    """Manual single-head attention

    Args:
        Q: [seq_len, d_k] Query matrix
        K: [seq_len, d_k] Key matrix
        V: [seq_len, d_v] Value matrix
        mask: Optional mask

    Returns:
        output: [seq_len, d_v] Attention output
        weights: [seq_len, seq_len] Attention weights
    """
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    scores = Q @ K.T / math.sqrt(d_k)

    # Step 2: Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax normalization
    weights = torch.softmax(scores, dim=-1)

    # Step 4: Weighted sum
    output = weights @ V

    return output, weights


# Experiment 2: Single-head attention computation
print("\n" + "=" * 60)
print("Experiment 2: Single-Head Attention Computation")
print("=" * 60)

seq_len = 4
d_k = 8

torch.manual_seed(42)
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

output, weights = single_head_attention(Q, K, V)

print(f"Q shape: {Q.shape} (4 words, each with 8-dim query)")
print(f"K shape: {K.shape} (4 words, each with 8-dim key)")
print(f"V shape: {V.shape} (4 words, each with 8-dim value)")
print(f"\nAttention weights shape: {weights.shape} (4x4, each row = one word's attention to others)")
print(f"Attention weights:\n{weights}")
print(f"\nRow sums: {weights.sum(dim=-1).tolist()} (each row sums to 1 after softmax)")
print(f"Output shape: {output.shape} (4 words' new representations)")


# ============================================================
# Step 3: Causal Mask — Language Models Can Only See the Past
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: Causal Mask")
print("=" * 60)

# Language models are autoregressive: when predicting token t, can only see tokens 0..t-1
# Implemented with lower-triangular mask:
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(f"Causal Mask (1=visible, 0=invisible):\n{causal_mask.int()}")

print("\nExplanation:")
print("  Position 0 can only see position 0")
print("  Position 1 can see positions 0,1")
print("  Position 2 can see positions 0,1,2")
print("  Position 3 can see positions 0,1,2,3")

# Attention with causal mask
output_causal, weights_causal = single_head_attention(Q, K, V, mask=causal_mask)
print(f"\nAttention weights with mask:\n{weights_causal}")
print("-> Upper triangle is all 0, each word only attends to itself and previous words")


# ============================================================
# Step 4: Multi-Head Attention
# ============================================================

print("\n" + "=" * 60)
print("Experiment 4: Multi-Head Attention — Attend from Different Angles")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    """Manual Multi-Head Attention Implementation"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        # Each head has its own Q, K, V projection
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape

        # Linear projection
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into multiple heads: [bsz, seq_len, hidden_size] -> [bsz, seq_len, num_heads, head_dim]
        Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim)
        K = K.view(bsz, seq_len, self.num_heads, self.head_dim)
        V = V.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Transpose: [bsz, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention computation
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores.float(), dim=-1).type_as(Q)
        output = weights @ V

        # Merge heads: [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        output = self.o_proj(output)

        return output, weights


# Experiment
hidden_size = 64
num_heads = 4
seq_len = 8
batch_size = 2

mha = MultiHeadAttention(hidden_size, num_heads)
x = torch.randn(batch_size, seq_len, hidden_size)
causal = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

output, weights = mha(x, mask=causal)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"  num_heads={num_heads}, head_dim={hidden_size//num_heads}")
print(f"\nEach head attends to different patterns:")
for h in range(num_heads):
    print(f"  Head {h}: position 0's weights for positions 0-3 = {weights[0, h, 0, :4].detach().tolist()}")


# ============================================================
# Step 5: GQA — Grouped Query Attention
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5: GQA — Grouped Query Attention")
print("=" * 60)

# Standard MHA: each Q head has independent K,V heads
#   8 Q heads -> 8 K heads + 8 V heads
# GQA: multiple Q heads share one set of K,V heads
#   8 Q heads -> 2 K heads + 2 V heads (every 4 Q heads share 1 KV group)

class GroupedQueryAttention(nn.Module):
    """GQA: Multiple Q heads share one set of KV heads"""

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads  # Q heads per KV head group
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape

        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Expand KV heads to match Q head count
        # [bsz, seq_len, num_kv_heads, head_dim] -> [bsz, seq_len, num_heads, head_dim]
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim)
            xk = xk.reshape(bsz, seq_len, self.num_heads, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim)
            xv = xv.reshape(bsz, seq_len, self.num_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores.float(), dim=-1).type_as(xq)
        output = weights @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        output = self.o_proj(output)
        return output, weights


# Compare MHA vs GQA
num_heads = 8
num_kv_heads_mha = 8
num_kv_heads_gqa = 2
head_dim = 16
hidden_size = num_heads * head_dim

mha_attn = MultiHeadAttention(hidden_size, num_heads)
gqa_attn = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads_gqa)

x = torch.randn(1, 8, hidden_size)

mha_params = sum(p.numel() for p in mha_attn.parameters())
gqa_params = sum(p.numel() for p in gqa_attn.parameters())

print(f"MHA: {num_heads} Q heads, {num_kv_heads_mha} KV heads, Parameters={mha_params:,}")
print(f"GQA: {num_heads} Q heads, {num_kv_heads_gqa} KV heads, Parameters={gqa_params:,}")
print(f"GQA parameter reduction: {(1 - gqa_params/mha_params)*100:.1f}%")
print(f"\nGQA Advantages:")
print(f"  1. Smaller KV cache (only need to store {num_kv_heads_gqa} groups instead of {num_kv_heads_mha})")
print(f"  2. Faster inference (KV cache reads reduced by {(1 - num_kv_heads_gqa/num_kv_heads_mha)*100:.0f}%)")
print(f"  3. Effectiveness close to MHA (shared KV heads still capture key information)")


# ============================================================
# Step 5.5: QK-Norm — Key Technique for Stable Attention Training
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5.5: QK-Norm — Key Technique for Stable Attention Training")
print("=" * 60)

print("""
Problem: When the model gets larger and training gets longer, Q·K^T values may explode
  -> softmax input too large -> output tends toward one-hot
  -> gradients vanish -> training becomes unstable or even crashes

Solution: Normalize Q and K (QK-Norm)
  Before computing attention scores, apply RMSNorm to each head's Q and K separately

  Standard Attention:  scores = Q @ K^T / sqrt(d_k)
  QK-Norm:            scores = RMSNorm(Q) @ RMSNorm(K)^T / sqrt(d_k)

MiniMind's original project uses QK-Norm, which is a standard technique in modern LLMs.
""")

class RMSNormForQK(nn.Module):
    """RMSNorm for QK-Norm (same as Lesson 3)"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class GroupedQueryAttentionWithQKNorm(nn.Module):
    """GQA with QK-Norm — consistent with MiniMind's original implementation"""

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = RMSNormForQK(self.head_dim)
        self.k_norm = RMSNormForQK(self.head_dim)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape

        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim)
            xk = xk.reshape(bsz, seq_len, self.num_heads, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim)
            xv = xv.reshape(bsz, seq_len, self.num_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores.float(), dim=-1).type_as(xq)
        output = weights @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        output = self.o_proj(output)
        return output, weights


gqa_with_qknorm = GroupedQueryAttentionWithQKNorm(hidden_size, num_heads, num_kv_heads_gqa)
gqa_without_qknorm = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads_gqa)

x = torch.randn(1, 8, hidden_size)
causal = torch.tril(torch.ones(8, 8)).unsqueeze(0).unsqueeze(0)

out_with, _ = gqa_with_qknorm(x, mask=causal)
out_without, _ = gqa_without_qknorm(x, mask=causal)

print(f"Without QK-Norm output range: [{out_without.min().item():.4f}, {out_without.max().item():.4f}]")
print(f"With QK-Norm output range: [{out_with.min().item():.4f}, {out_with.max().item():.4f}]")

qknorm_extra = sum(p.numel() for p in gqa_with_qknorm.q_norm.parameters()) + sum(p.numel() for p in gqa_with_qknorm.k_norm.parameters())
print(f"\nQK-Norm extra parameters: {qknorm_extra} (head_dim x 2 = {hidden_size // num_heads} x 2)")
print("-> Minimal parameter increase, but training stability significantly improved")


# ============================================================
# Step 6: MiniMind's Attention Configuration
# ============================================================

print("\n" + "=" * 60)
print("Experiment 6: MiniMind's Attention Configuration")
print("=" * 60)

configs = {
    "MiniMind (Small)": {"num_heads": 12, "num_kv_heads": 4, "head_dim": 64, "hidden_size": 768},
    "MiniMind (Medium)": {"num_heads": 16, "num_kv_heads": 4, "head_dim": 64, "hidden_size": 1024},
}

for name, cfg in configs.items():
    n_rep = cfg["num_heads"] // cfg["num_kv_heads"]
    kv_cache_per_token = cfg["num_kv_heads"] * cfg["head_dim"] * 2  # K+V
    print(f"{name}:")
    print(f"  Q heads: {cfg['num_heads']}, KV heads: {cfg['num_kv_heads']}, each group of {n_rep} Q heads shares 1 KV group")
    print(f"  KV cache per token: {kv_cache_per_token} floats = {kv_cache_per_token*4/1024:.1f} KB")
    print(f"  KV cache for 2048 tokens: {kv_cache_per_token*2048*4/1024/1024:.2f} MB")
    print()


# ============================================================
# Step 7: Complete Attention Computation Flow
# ============================================================

print("=" * 60)
print("Experiment 7: Complete Flow from Input to Output")
print("=" * 60)

# Using MiniMind small model config
hidden_size = 768
num_heads = 12
num_kv_heads = 4
head_dim = 64
seq_len = 16
batch_size = 1

# Create components
q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

# Simulated input (after Embedding + RMSNorm)
x = torch.randn(batch_size, seq_len, hidden_size)

# Step 1: Linear projection
xq = q_proj(x)
xk = k_proj(x)
xv = v_proj(x)
print(f"1. Linear projection: x[{x.shape}] -> Q[{xq.shape}], K[{xk.shape}], V[{xv.shape}]")

# Step 2: Reshape to multi-head format
xq = xq.view(batch_size, seq_len, num_heads, head_dim)
xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim)
xv = xv.view(batch_size, seq_len, num_kv_heads, head_dim)
print(f"2. Reshape: Q[{xq.shape}], K[{xk.shape}], V[{xv.shape}]")

# Step 3: Expand KV heads (GQA)
n_rep = num_heads // num_kv_heads
xk_exp = xk[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, num_heads, head_dim)
xv_exp = xv[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, num_heads, head_dim)
print(f"3. GQA expansion: K[{xk_exp.shape}], V[{xv_exp.shape}]")

# Step 4: Transpose to [batch, heads, seq, dim]
xq = xq.transpose(1, 2)
xk_exp = xk_exp.transpose(1, 2)
xv_exp = xv_exp.transpose(1, 2)
print(f"4. Transpose: Q[{xq.shape}], K[{xk_exp.shape}], V[{xv_exp.shape}]")

# Step 5: Attention scores
scores = xq @ xk_exp.transpose(-2, -1) / math.sqrt(head_dim)
print(f"5. Attention scores: [{scores.shape}]")

# Step 6: Causal mask
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
scores_masked = scores + causal_mask
print(f"6. Apply causal mask: upper triangle -> -inf")

# Step 7: Softmax
attn_weights = torch.softmax(scores_masked.float(), dim=-1)
print(f"7. Softmax: attention weights [{attn_weights.shape}]")

# Step 8: Weighted sum
attn_output = attn_weights @ xv_exp
print(f"8. Weighted sum: [{attn_output.shape}]")

# Step 9: Merge heads + output projection
attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
output = o_proj(attn_output)
print(f"9. Merge heads + projection: [{output.shape}]")

print(f"\nInput: [{batch_size}, {seq_len}, {hidden_size}]")
print(f"Output: [{output.shape}]")
print("Attention doesn't change shape, only changes content!")


# ============================================================
# Step 8: Attention Weight Visualization
# ============================================================

print("\n" + "=" * 60)
print("Experiment 8: Attention Weight Patterns")
print("=" * 60)

# Use a simple example to observe attention patterns
seq_len = 6
d_k = 4

# Simulate attention for a sentence: "The cat sat on the mat"
words = ["The", "cat", "sat", "on", "the", "mat"]

torch.manual_seed(7)
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

causal = torch.tril(torch.ones(seq_len, seq_len))
_, weights = single_head_attention(Q, K, V, mask=causal)

print(f"Sentence: {' '.join(words)}")
print(f"\nAttention weight matrix (each row = one word's attention to others):")
print(f"{'':>6}", end="")
for w in words:
    print(f"{w:>6}", end="")
print()

for i, w in enumerate(words):
    print(f"{w:>6}", end="")
    for j in range(seq_len):
        val = weights[i, j].item()
        if val > 0.3:
            print(f"  ***", end="")
        elif val > 0.15:
            print(f"   **", end="")
        elif val > 0.05:
            print(f"    *", end="")
        else:
            print(f"    .", end="")
    print()

print("\n*** = High attention, ** = Medium attention, * = Low attention, . = Almost none")
print("Causal mask ensures each word only sees itself and previous words (lower triangle)")


# ============================================================
# Deep Dive: Attention Intuition and Analogies
# ============================================================
print("\n" + "=" * 60)
print("Deep Dive: Attention Intuition and Analogies")
print("=" * 60)

print("""
[Analogy 1: Library Book Search]
--------------------------------
Imagine you're in a library looking for books:
  - Q (Query)   = your "question" or "interest" (e.g. "I want to learn Python")
  - K (Key)     = each book's "label" or "index" (title, keywords)
  - V (Value)   = each book's "actual content"

  Process:
    1. Compare your Q with all books' K (dot product)
    2. Books with high similarity get high weights (softmax)
    3. Mix all books' content V according to weights
    4. The mixed result is "the knowledge you want"

  -> Attention is "weighted sum by relevance"


[Analogy 2: Meeting Discussion]
-------------------------------
In a meeting, everyone speaks, but you only listen carefully to relevant people:
  - Q = the topic you care about
  - K = each person's speaking topic
  - V = each person's detailed content
  - Dot product = "how relevant is their speech to my topic"
  - Softmax = decides "how much attention to give them"

  In a Transformer:
    Each word is a "speaker"
    It's both a listener (uses Q to query)
    And a speaker (contributes content via V)


[Diagram: Attention Computation Flow]
--------------------------------------

   Input: Q, K, V
       |
       v
   +----------------+
   | Q @ K^T        |  <- Compute similarity
   +-------+--------+
           |
           v
   +----------------+
   | / sqrt(d_k)    |  <- Scale, prevent softmax saturation
   +-------+--------+
           |
           v
   +----------------+
   | + Mask (opt)   |  <- Causal mask
   +-------+--------+
           |
           v
   +----------------+
   | Softmax (row)  |  <- Normalize to probabilities
   +-------+--------+
           |
           v
   +----------------+
   | @ V            |  <- Weighted sum
   +-------+--------+
           |
           v
       Output

Each row's (each query's) weights sum to 1
  -> Each position "distributes 1 unit of attention" across all positions


[Why divide by sqrt(d_k)?]
---------------------------
  Problem: When d_k is large, Q·K^T variance also increases
  Large variance -> softmax tends toward one-hot (only attends to one position)
  -> Small gradients, difficult training

  Mathematically:
    Q, K elements ~ N(0, 1) (assumed)
    Q·K^T = sum_i Q_i * K_i
    Var(Q·K^T) = d_k
    Std = sqrt(d_k)
    -> After dividing by sqrt(d_k), variance is normalized to 1

  Experiment: d_k=64, without scaling, softmax output is nearly one-hot
              d_k=64, with scaling, softmax output is smoother

[Geometric Meaning of Multi-Head Attention]
--------------------------------------------
Each head learns different "attention patterns":
  - Head 1: subject-verb relationships (verb looks at subject)
  - Head 2: coreference (pronoun looks at noun)
  - Head 3: long-distance dependencies (beginning-end connections)
  - Head 4: local syntax (adjacent words)
  - ...

  Analogy: A team working together, each person specializes in one aspect
  -> Teamwork is more comprehensive than working alone!
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Source of Q, K, V
  Q, K, V all come from input x. Why do 3 different linear projections?

[Exercise 2] Scaling Factor
  Attention divides by sqrt(d_k). What happens if we don't?

[Exercise 3] Causal Mask
  When generating token t, can it see content before or after position t?

[Exercise 4] Multi-Head vs Single-Head
  With the same d-dimensional vector, what advantages does multi-head attention have?

[Exercise 5] Attention Weight Sum
  What is the sum of each row in softmax(QK^T)? What's its physical meaning?

[Exercise 6] Self-Attention vs Cross-Attention
  Self-attention: Q, K, V all come from the same input
  Cross-attention: Q comes from one input, K and V from another
  What scenarios is each used in?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Q, K, V are 3 different 'views' of the same vector")
print("  - W_Q projection: learns 'what this token is looking for' (query)")
print("  - W_K projection: learns 'what this token can be queried by' (key)")
print("  - W_V projection: learns 'what information this token wants to convey' (value)")
print()
print("  If using the same projection (Q=K=V=x), attention degenerates to:")
print("  softmax(x·x^T)·x, lacking 'question-answer' decoupling")
print("  -> Model cannot distinguish 'what I'm looking for' from 'what I can provide'")
print("  -> Same logic for multi-head: each head learns different W_Q, W_K, W_V")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Without dividing by sqrt(d_k):")
print("  - With large d_k, QK^T values become too large")
print("  - Softmax tends toward one-hot (only attends to one position)")
print("  - Gradients vanish (other positions' gradients approach 0)")
print("  - Training becomes unstable, model struggles to learn diverse attention")
print()
print("  Demo:")
for d_k in [4, 16, 64, 256]:
    q = torch.randn(1, 1, 1, d_k)
    k = torch.randn(1, 1, 5, d_k)
    scores = (q @ k.transpose(-2, -1)).squeeze()
    print(f"    d_k={d_k}: unscaled scores={scores.tolist()}")
    weights = F.softmax(scores, dim=-1)
    print(f"            softmax={weights.tolist()}")
    print(f"            max={weights.max():.4f} (larger = more one-hot)")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Can only see content before position t (including itself)")
print("  Causal mask sets upper triangle to -inf")
print("  softmax(-inf) = 0, future positions don't participate in attention")
print()
print("  Why? Language models are autoregressive")
print("  During training, the full sentence is known, but during inference, tokens are generated one by one")
print("  If the model sees the future during training, it won't know the future during inference")
print("  -> Training-inference mismatch -> training fails")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  Multi-head attention advantages:")
print("  1. Attend to different aspects: one head focuses on syntax, another on semantics")
print("  2. Attend to different positions: one head looks nearby, another looks far away")
print("  3. Attend to different relationships: subject-verb, verb-object, coreference, etc.")
print()
print("  Mathematically: multi-head splits the d-dim vector into h parts, each d/h dims")
print("  Each head independently computes attention, then concatenates results")
print("  -> Equivalent to attention in low-rank subspaces, more expressive")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Each row sums to 1")
print("  Softmax normalizes each row to a probability distribution")
print()
print("  Physical meaning: 'I distribute 1 unit of attention across all positions'")
print("  - Self = 0.5 (mainly attend to self)")
print("  - Key word = 0.3 (important content)")
print("  - Irrelevant words = 0.1 + 0.05 + 0.05 = 0.2 (background)")
print("  -> Total = 1.0")
print()
print("  This 'probability' interpretation makes attention very elegant")
print("  You can also visualize attention maps to understand what the model focuses on")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Self-Attention:")
print("    Q, K, V all come from the same sequence")
print("    Use: let information flow within the sequence")
print("    Example: GPT (text generation), BERT (text understanding)")
print()
print("  Cross-Attention:")
print("    Q comes from one sequence, K, V from another sequence")
print("    Use: let one sequence 'query' another sequence")
print("    Example: Translation (Chinese Q queries English K, V)")
print("    Example: Text-to-image (text Q queries image K, V)")
print("    Example: RAG (question Q queries document K, V)")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
print("""
1. Attention = softmax(Q·K^T/sqrt(d))·V, lets words exchange information
2. Q=Query, K=Key, V=Value, similar to information retrieval
3. Causal mask ensures language models can only see the past, not the future
4. Multi-head attention lets the model attend from different angles
5. GQA lets multiple Q heads share KV heads, reducing KV cache overhead
6. MiniMind uses GQA (e.g. 12 Q heads share 4 KV heads)
7. QK-Norm normalizes Q and K, preventing attention score explosion, stabilizing training

Data flow so far:
  Token IDs -> Embedding -> RMSNorm -> RoPE -> Attention -> ...
  [42,108]    [0.1,-0.3]   [0.8,0.2]  rotated Q/K  context-aware vectors

Next -> lesson06_ffn.py: Feed-Forward Network — memory refinement after attention
""")
