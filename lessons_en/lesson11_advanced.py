"""
Advanced Lesson: MiniMind's Advanced Features
==============================================

In the basic lessons, we learned the core components of GPT. Now let's explore MiniMind's advanced features:

  1. MoE (Mixture of Experts): Different experts handle different content
  2. TTT (Test-Time Training): Model evolves during inference
  3. MTP (Multi-Token Prediction): Predict multiple future tokens at once
  4. KV Cache: Key optimization for accelerating generation
  5. MSA (MiniMax Sparse Attention): Sparse attention for ultra-long texts

Run: python lessons_en/lesson11_advanced.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. MoE — Mixture of Experts
# ============================================================

print("=" * 60)
print("[1] MoE — Mixture of Experts: Different experts handle different content")
print("=" * 60)

class SimpleExpert(nn.Module):
    """A single expert: a standard SwiGLU FFN"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleMoE(nn.Module):
    """Simplified MoE: Gating router + multiple experts

    Consistent with MiniMind's MOEFeedForward, including:
    - norm_topk_prob: Normalize Top-K weights (ensure weights sum to 1)
    - aux_loss: Load balancing auxiliary loss (prevent all tokens from being routed to the same expert)
    """

    def __init__(self, hidden_size, intermediate_size, num_experts=4, top_k=2,
                 norm_topk_prob=True, router_aux_loss_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        self.experts = nn.ModuleList([
            SimpleExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)

        gate_scores = F.softmax(self.gate(x_flat), dim=-1)

        topk_weight, topk_idx = torch.topk(gate_scores, k=self.top_k, dim=-1, sorted=False)

        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * gate_scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = gate_scores.new_zeros(1).squeeze()

        return y.reshape(bsz, seq_len, hidden)


# Experiment
hidden_size = 64
intermediate_size = 128

moe = SimpleMoE(hidden_size, intermediate_size, num_experts=4, top_k=2)
ffn = SimpleExpert(hidden_size, intermediate_size)

x = torch.randn(2, 8, hidden_size)
out_moe = moe(x)
out_ffn = ffn(x)

moe_params = sum(p.numel() for p in moe.parameters())
ffn_params = sum(p.numel() for p in ffn.parameters())

print(f"Standard FFN parameters: {ffn_params:,}")
print(f"MoE (4 experts, Top-2) parameters: {moe_params:,} ({moe_params/ffn_params:.1f}x)")
print(f"But each inference only uses Top-2 experts, actual computation ≈ {2/4*100:.0f}%")
print(f"\nCore idea of MoE:")
print(f"  - Large parameter count → rich knowledge")
print(f"  - Only use some experts each time → controllable computation")
print(f"  - Gating network automatically selects the most suitable experts")
print(f"  - Different types of tokens are routed to different experts")

# [NEW] Router Collapse Problem and aux_loss Solution
print("\n" + "-" * 50)
print("[MoE's Router Collapse Problem & Solution]")
print("-" * 50)
print("""
Problem: Without aux_loss, the router tends to send all tokens to the same expert.

  Why? The "rich get richer" effect:
    1. Expert 0 gets slightly more tokens initially (random)
    2. Expert 0 gets more training → becomes better
    3. Better expert → router sends it more tokens
    4. Other experts get less training → become worse
    5. Eventually: Expert 0 handles everything, others are wasted

  Without aux_loss (load distribution):
    Expert 0: ████████████████████ 80% of tokens
    Expert 1: ████ 15% of tokens
    Expert 2: █ 3% of tokens
    Expert 3: ▏ 2% of tokens
    → 3 experts are wasted!

  With aux_loss (load distribution):
    Expert 0: █████ 27% of tokens
    Expert 1: ████ 25% of tokens
    Expert 2: █████ 26% of tokens
    Expert 3: ████ 22% of tokens
    → All experts contribute!

  How aux_loss works:
    aux_loss = (load × gate_scores.mean(0)).sum() × num_experts × coef
    - load = fraction of tokens each expert receives
    - gate_scores.mean(0) = average routing probability per expert
    - When load is uneven, this loss is high → encourages even distribution
    - coef (typically 0.01) controls how strongly we enforce balance
""")


# ============================================================
# 2. TTT — Test-Time Training
# ============================================================

print("\n" + "=" * 60)
print("[2] TTT — Test-Time Training: Model evolves during inference")
print("=" * 60)

print("""
Core idea of TTT (In-Place Test-Time Training):

  Standard model: Parameters are fixed after training, not updated during inference
  TTT model: Fine-tune parameters during inference to adapt to current context

  Specific approach:
  1. Perform mini-batch SGD on FFN's down_proj
  2. Use next-token prediction as self-supervised signal
  3. Update a few steps each inference, compressing context info into weights

  Analogy:
    Standard model = Can only use existing knowledge during an exam
    TTT model = Can read books during the exam (but only content relevant to the current question)

  MiniMind's TTT implementation:
  - Only update FFN's down_proj
  - Use a lightweight linear head to build next-token prediction signal
  - Reset weights every chunk (e.g., 512 tokens)
  - Very small learning rate (1e-4) to avoid excessive modification
""")

# Simplified TTT demo
class SimpleTTTLayer(nn.Module):
    """Simplified TTT: Fine-tune down_proj during inference"""

    def __init__(self, hidden_size, intermediate_size, ttt_lr=1e-4):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.ttt_lr = ttt_lr
        self.ttt_predictor = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, use_ttt=False):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        if use_ttt and not self.training:
            # TTT: Fine-tune down_proj during inference
            W = self.down_proj.weight.clone()
            for t in range(x.shape[1] - 1):
                # Output at current position
                out = F.linear(hidden[:, t], W)
                # Self-supervision: Use output to predict next position's input
                pred = self.ttt_predictor(out)
                target = x[:, t + 1]
                loss = F.mse_loss(pred, target.detach())
                # Compute gradient and update
                grad = torch.autograd.grad(loss, W, retain_graph=False)[0]
                W = W - self.ttt_lr * grad

            # Use updated weights for final output
            output = F.linear(hidden, W)
        else:
            output = self.down_proj(hidden)

        return output

ttt_layer = SimpleTTTLayer(64, 128)
x = torch.randn(1, 8, 64)

out_normal = ttt_layer(x, use_ttt=False)
out_ttt = ttt_layer(x, use_ttt=True)

print(f"Standard inference output std: {out_normal.std():.4f}")
print(f"TTT inference output std:  {out_ttt.std():.4f}")
print(f"TTT allows the model to adapt to input during inference, output may differ")


# ============================================================
# 3. MTP — Multi-Token Prediction
# ============================================================

print("\n" + "=" * 60)
print("[3] MTP — Multi-Token Prediction: Predict multiple future tokens at once")
print("=" * 60)

print("""
Standard language model: Predict only the next token each time
MTP (Multi-Token Prediction): Simultaneously predict the next N tokens

Why MTP?
  1. Richer training signal: Each position learns N targets instead of 1
  2. Better planning ability: Model must "think ahead" for several steps
  3. Can generate multiple tokens in parallel during inference, speeding up

Implementation:
  Append N prediction heads after lm_head:
    head_0: Predict token at position t+1 (standard)
    head_1: Predict token at position t+2
    head_2: Predict token at position t+3
    ...

  Each prediction head has its own Norm + Linear + Residual
""")

class SimpleMTPHead(nn.Module):
    """An MTP prediction head"""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x, base_logits):
        # Residual connection to base logits
        projected = self.proj(self.norm(x))
        return base_logits + self.lm_head(projected)


# Demo
vocab_size = 200
hidden_size = 64
num_mtp_heads = 3

base_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
mtp_heads = nn.ModuleList([
    SimpleMTPHead(hidden_size, vocab_size)
    for _ in range(num_mtp_heads)
])

x = torch.randn(1, 8, hidden_size)
base_logits = base_lm_head(x)

print(f"Base prediction head: Predict position t+1")
for i, head in enumerate(mtp_heads):
    logits = head(x, base_logits)
    print(f"MTP head {i+1}: Predict position t+{i+2}, logits shape={logits.shape}")

print(f"\nDuring training: Weighted sum of losses from 4 prediction heads")
print(f"  total_loss = loss_head0 + 0.1 * (loss_head1 + loss_head2 + loss_head3)")
print(f"  MTP auxiliary loss weight is usually small (0.1) to avoid interfering with main prediction")


# ============================================================
# 4. KV Cache — Key optimization for accelerating generation
# ============================================================

print("\n" + "=" * 60)
print("[4] KV Cache — Avoid redundant computation")
print("=" * 60)

class SimpleKVCache:
    """Simplified KV Cache"""

    def __init__(self, num_layers, max_len, num_kv_heads, head_dim, device='cpu'):
        self.max_len = max_len
        self.k_cache = [torch.zeros(1, max_len, num_kv_heads, head_dim, device=device) for _ in range(num_layers)]
        self.v_cache = [torch.zeros(1, max_len, num_kv_heads, head_dim, device=device) for _ in range(num_layers)]
        self.cur_len = [0] * num_layers

    def update(self, layer_idx, new_k, new_v):
        cur = self.cur_len[layer_idx]
        new_len = cur + new_k.shape[1]
        self.k_cache[layer_idx][:, cur:new_len].copy_(new_k)
        self.v_cache[layer_idx][:, cur:new_len].copy_(new_v)
        self.cur_len[layer_idx] = new_len
        return self.k_cache[layer_idx][:, :new_len], self.v_cache[layer_idx][:, :new_len]


# Compare computation with and without KV Cache
import time

class SimpleModelWithKVCache(nn.Module):
    """Simplified model supporting KV Cache"""

    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*4,
            batch_first=True, dropout=0.0
        ) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

model_cache = SimpleModelWithKVCache(200, 64)
model_cache.eval()

# Naive generation
input_ids = torch.randint(0, 200, (1, 4))
start = time.time()
generated = input_ids.clone()
with torch.no_grad():
    for _ in range(20):
        logits = model_cache(generated)
        next_tok = logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_tok], dim=1)
time_no_cache = (time.time() - start) * 1000

# Count computation
total_tokens_processed = sum(4 + i for i in range(20))
print(f"Naive generation of 20 tokens:")
print(f"  Total tokens processed in forward pass: {total_tokens_processed}")
print(f"  Time: {time_no_cache:.1f} ms")
print(f"\nKV Cache generation:")
print(f"  Step 1: Process 4 tokens (prefill)")
print(f"  Steps 2-20: Process only 1 token each (decode)")
print(f"  Total processed: 4 + 19 = 23 tokens (vs {total_tokens_processed})")
print(f"  Savings: {(1 - 23/total_tokens_processed)*100:.0f}% computation")


# ============================================================
# 5. MSA — Sparse Attention
# ============================================================

print("\n" + "=" * 60)
print("[5] MSA — Sparse Attention: Processing ultra-long texts")
print("=" * 60)

print("""
Problem with standard Attention:
  Computation = O(N^2), where N is sequence length
  N=1024 → 1M operations
  N=8192 → 67M operations → Too slow!

MSA (MiniMax Sparse Attention) solution:
  Not every token attends to all other tokens, only the "important" ones

  Two-stage design:
  ┌─────────────────────────────────────┐
  │  Stage 1: Index Branch              │
  │  - Use low-dim projection to compute│
  │    relevance scores for each block   │
  │  - Select Top-K most important blocks│
  │  - Low computation, fast filtering   │
  └─────────────────────────────────────┘
                 ↓ Selected block indices
  ┌─────────────────────────────────────┐
  │  Stage 2: Sparse Branch             │
  │  - Full attention only on selected   │
  │    blocks                            │
  │  - Skip irrelevant blocks, save comp │
  │  - GQA groups share indices, further │
  │    reducing overhead                 │
  └─────────────────────────────────────┘

  Short sequences (< fallback_len): Automatically falls back to standard attention
  Long sequences: Only compute top-k proportion of blocks, complexity O(N * top_k * block_size)
""")

# Demo block-level selection
seq_len = 128
block_size = 16
n_blocks = seq_len // block_size
topk_ratio = 0.25
topk = max(1, int(n_blocks * topk_ratio))

print(f"Sequence length: {seq_len}, Block size: {block_size}")
print(f"Total blocks: {n_blocks}, Top-K: {topk} (ratio={topk_ratio})")

# Simulate index branch block selection
torch.manual_seed(42)
block_scores = torch.randn(1, seq_len, 1, n_blocks)
_, topk_indices = torch.topk(block_scores, k=topk, dim=-1)

print(f"\nEach position only attends to {topk}/{n_blocks} blocks = {topk/n_blocks*100:.0f}%")
print(f"Standard attention: Each position attends to {seq_len} tokens")
print(f"MSA attention: Each position attends to ~{topk * block_size} tokens")
print(f"Computation savings: ~{(1 - topk/n_blocks)*100:.0f}%")

print(f"\nKey design choices of MSA:")
print(f"  1. Block-level selection (block_size=64): Coarse-grained filtering, reduces index overhead")
print(f"  2. GQA group-shared indices: Q heads in the same group share selection results")
print(f"  3. Fallback mechanism: Short sequences automatically fall back to standard attention")
print(f"  4. Causal mask: Ensure only past blocks are visible")


# ============================================================
# Advanced Lesson Summary
# ============================================================

print("\n" + "=" * 60)
print("Advanced Lesson Summary")
print("=" * 60)
print("""
MiniMind's 5 advanced features:

1. MoE (Mixture of Experts)
   - Large parameter count but controllable computation
   - Gating network for automatic routing
   - Suitable for scaling model size without increasing inference cost

2. TTT (Test-Time Training)
   - Fine-tune parameters during inference to adapt to context
   - Self-supervised signal: next-token prediction
   - Allows model to "learn" during inference

3. MTP (Multi-Token Prediction)
   - Simultaneously predict the next N tokens
   - Richer training signal, stronger planning ability
   - Can generate in parallel during inference

4. KV Cache
   - Cache computed KV to avoid redundant computation
   - Generation speed reduced from O(N^2) to O(N)
   - Pre-allocated buffers to avoid dynamic concatenation

5. MSA (Sparse Attention)
   - Two stages: Index branch filtering + Sparse branch computation
   - Significantly reduced computation for long sequences
   - Short sequences automatically fall back to standard attention

These features enable MiniMind to achieve powerful capabilities at a small model scale!
Congratulations on completing all MiniMind learning lessons!
""")


# ============================================================
# Deep Understanding: Comparison of Five Advanced Features
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: Comparison of Five Advanced Features")
print("=" * 60)

print("""
[Analogy: LLM's "Superpowers"]
─────────────────────
  MoE    = Multiple expert consultations, use whichever is relevant
  TTT    = Remember after reading once, take notes during inference
  MTP    = Think three steps ahead, plan in advance
  KV     = Bookmark pages you've already read, no need to flip back
  MSA    = Find key points, quickly skim unimportant content


[What problems do the five features solve?]
──────────────────────

  MoE (Mixture of Experts):
    Problem: Bigger model = More computation
    Solution: Only activate a few experts each time, large capacity but low computation
    Analogy: Large hospital, patients are triaged, not all doctors need to see every patient
    Key: Gating network (router) decides which expert handles each token

  TTT (Test-Time Training):
    Problem: Context information is hard to incorporate into the model
    Solution: Still "learning" during inference (fine-tune down_proj)
    Analogy: Taking notes while reading, understanding improves as you read
    Key: No additional parameters, only temporary adjustments during inference

  MTP (Multi-Token Prediction):
    Problem: Predicting one token at a time is slow for training
    Solution: Look multiple steps ahead, predict next k tokens simultaneously
    Analogy: Learning chess, think three moves ahead
    Key: Multiple output heads + shared backbone

  KV Cache:
    Problem: Repeatedly computing K, V during generation
    Solution: Store K, V, compute once and reuse
    Analogy: Keep reference book open during exam, no need to close and reopen
    Key: Pre-allocated buffer, avoid dynamic concatenation

  MSA (MiniMax Sparse Attention):
    Problem: Attention computation grows quadratically with sequence length
    Solution: Only compute full attention for important positions
    Analogy: Reading comprehension, carefully read key paragraphs, skim the rest
    Key: Index selects top-k blocks, sparse computation


[Diagram: KV Cache vs MSA Difference]
─────────────────────────

  KV Cache (Accelerates generation, doesn't change computation):
    
    Without cache:
      Step 1: [token1, token2, token3]           → Compute K1, V1, K2, V2, K3, V3
      Step 2: [token1, token2, token3, t1_new]   → Compute K1, V1, K2, V2, K3, V3, K4, V4  ← Redundant!
      Step 3: [token1, ..., t1_new, t2_new]      → Compute K1, V1, ..., K4, V4, K5, V5   ← Redundant!
    
    With cache:
      Step 1: [token1, token2, token3]           → Compute K1, V1, K2, V2, K3, V3
      Step 2: [t1_new] (only new)                → Compute K4, V4, reuse previous
      Step 3: [t2_new]                           → Compute K5, V5, reuse previous

  MSA (Changes attention computation, reduces complexity):
    
    Standard attention:
      Q K^T → Compute similarity for all position pairs
      O(N^2) computation, N=4096 → 16M elements
    
    Sparse attention:
      Q K^T → Only compute full attention for top-k important positions
      O(N * k) computation, k=512 → 2M elements
      8x speedup!

  → KV Cache solves "redundant computation", applicable to all Transformers
  → MSA solves "full attention waste", applicable to long sequences


[When to enable these features?]
──────────────────

  Model scale:
    Small (< 1B):  No need for MoE, KV Cache is sufficient
    Medium (1B-7B): Add KV Cache, MTP training, MSA
    Large (> 7B):  MoE, TTT, MSA all enabled

  Task:
    Long text (> 4K): Must enable MSA, KV Cache
    Multi-task: MoE lets different tasks use different experts
    Reasoning enhancement: TTT lets model learn online

  Deployment environment:
    Tight VRAM: KV Cache quantization, MoE offloading
    Tight speed: MSA, KV Cache both needed
    Tight accuracy: TTT, MTP

  MiniMind current status:
    All five features implemented, can be toggled via configuration
    Users can choose based on hardware and task


[Comprehensive Experiment: Observe Effects of Each Feature]
─────────────────────────
""")

# Simple experiment: Compare with and without KV Cache
import time

print("Experiment 1: KV Cache Speedup Effect")
print("-" * 40)

seq_lens_to_test = [64, 128, 256]
print(f"{'Seq Len':<12} {'No Cache (ms)':<16} {'With Cache (ms)':<16} {'Speedup':<10}")
print("-" * 60)

# Simulate simple Q @ K^T
for seq_len in seq_lens_to_test:
    Q = torch.randn(1, 8, seq_len, 64)
    K = torch.randn(1, 8, seq_len, 64)
    V = torch.randn(1, 8, seq_len, 64)

    # Without cache
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(5):
        attn = torch.matmul(Q, K.transpose(-2, -1))
        out = torch.matmul(attn, V)
    t1 = time.time()
    no_cache_time = (t1 - t0) * 100

    # With cache: Only compute new token
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(5):
        q_new = Q[:, :, -1:]
        attn = torch.matmul(q_new, K.transpose(-2, -1))
        out = torch.matmul(attn, V)
    t1 = time.time()
    cache_time = (t1 - t0) * 100

    speedup = no_cache_time / max(cache_time, 1e-6)
    print(f"{seq_len:<12} {no_cache_time:<16.2f} {cache_time:<16.2f} {speedup:<10.2f}x")

print("\nExperiment 2: Sparse Attention vs Full Attention (Theoretical)")
print("-" * 40)

for N in [512, 1024, 2048, 4096, 8192]:
    full = N * N
    sparse = N * 256  # top-256
    saving = (1 - sparse / full) * 100
    print(f"  N={N:5d}: Full attention {full**1:>14,} → Sparse {sparse:>9,} (Saves {saving:.1f}%)")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] MoE Routing
  8 experts, top-2 routing, how many experts are activated per inference? How much computation is saved?

[Exercise 2] TTT vs Fine-tuning
  What is the essential difference between TTT and traditional fine-tuning?

[Exercise 3] MTP Advantage
  How much additional loss does MTP have compared to single-token prediction? When k=4

[Exercise 4] KV Cache Memory
  Generating 4096 tokens, batch=1, 32 layers, 8 KV heads, head_dim=128
  How much memory does KV Cache occupy? (FP16)

[Exercise 5] MSA Fallback
  Why does MSA fall back to standard attention for short sequences (N=128)?

[Exercise 6] Feature Combination
  Which features affect loss during training? Which affect latency during inference?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  8 experts, top-2 routing:")
print("    Activated: 2 experts")
print("    Saved: (8-2)/8 = 75% computation")
print()
print("  Actual computation (relative):")
print("    All experts activated: 8x")
print("    top-2 activated: 2x")
print("    Saved: 6x (75% off)")
print()
print("  Further considerations:")
print("    - Gating network itself has computation (8 logits → softmax → top-2)")
print("    - But it's small relative to expert computation")
print("    - Net savings ~70%")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Essential difference:")
print()
print("  Traditional fine-tuning:")
print("    - Large data + large compute + update all parameters after training")
print("    - After fine-tuning, model is fixed, no longer changes")
print("    - Analogy: Intensive training during winter/summer break")
print()
print("  TTT:")
print("    - Fine-tune a small number of parameters online during inference for current input")
print("    - Only modify down_proj (1 matrix)")
print("    - Not persisted, forgets when input changes")
print("    - Analogy: Looking up references while treating a patient")
print()
print("  Key differences:")
print("    - Fine-tuning: Offline, persistent, all parameters")
print("    - TTT: Online, temporary, local parameters")
print()
print("  Practical applications:")
print("    TTT suitable for: Personalized dialogue, long document Q&A")
print("    Fine-tuning suitable for: General capability improvement, style customization")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  MTP k=4, has 4x prediction loss compared to single-token")
print()
print("  Loss composition:")
print("    L_total = L_1 + L_2 + L_3 + L_4")
print("    L_i = Cross-entropy for predicting the i-th future token")
print()
print("  In practice:")
print("    L_1 = Direct next token loss")
print("    L_2, L_3, L_4 = Indirect supervision")
print("    → Main loss is still L_1, others are auxiliary")
print()
print("  During training:")
print("    L_total = L_1 + 0.5 * (L_2 + L_3 + L_4)  (weighted)")
print("    → Auxiliary loss weight should be lower to avoid overshadowing the main loss")
print()
print("  During inference:")
print("    MTP heads don't participate, inference speed same as single-token")

# Exercise 4
print("\n[Exercise 4 Answer]")
seq_len = 4096
batch = 1
num_layers = 32
num_kv_heads = 8
head_dim = 128
fp16_bytes = 2

total_elements = 2 * batch * num_layers * seq_len * num_kv_heads * head_dim
total_bytes = total_elements * fp16_bytes
total_mb = total_bytes / (1024 * 1024)
total_gb = total_mb / 1024

print(f"  Config: bs={batch}, layers={num_layers}, seq={seq_len}")
print(f"  num_kv_heads={num_kv_heads}, head_dim={head_dim}")
print(f"  K, V each: {batch} x {num_layers} x {seq_len} x {num_kv_heads} x {head_dim}")
print(f"  K+V total elements: {total_elements:,}")
print(f"  Bytes: {total_bytes:,} = {total_mb:.1f} MB = {total_gb:.2f} GB")
print()
print(f"  → Single sequence with 4K context, KV Cache already 0.5 GB")
print(f"  → This is why long context needs KV Cache quantization")
print(f"  → PagedAttention (vLLM) further optimizes through paging")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Reasons for short sequence fallback:")
print()
print("  1. Sparse benefit is too small:")
print("    N=128, top-k=64 → Saves 50%")
print("    N=4096, top-k=64 → Saves 98%")
print("    For short sequences, full attention constant is small, sparse actually introduces overhead")
print()
print("  2. Index branch overhead is fixed:")
print("    Index branch needs to compute N block scores")
print("    When N is small, this overhead proportion is actually larger")
print()
print("  3. Fallback strategy:")
print("    if N < threshold:  # Usually threshold=512-1024")
print("        use standard attention")
print("    else:")
print("        use sparse attention")
print()
print("  MiniMind implementation:")
print("    Has fallback mechanism, short sequences automatically use standard attention")
print("    Avoiding overhead from sparsity")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Features affecting loss during training:")
print("    - MTP: Adds auxiliary loss, direct impact")
print("    - MoE: Affects routing distribution, indirect impact")
print("    - TTT: Only used during inference")
print("    - KV Cache: Not used in training, for acceleration")
print("    - MSA: Can be used in both training and inference")
print()
print("  Features affecting latency during inference:")
print("    - KV Cache: Greatly reduces latency (Nx speedup)")
print("    - MSA: Reduces latency (sparse computation)")
print("    - MoE: Increases latency (routing overhead) but enables larger models")
print("    - TTT: Increases latency (training during inference)")
print("    - MTP: Not involved in inference, no impact")
print()
print("  Practical deployment choices:")
print("    Speed priority: KV Cache + MSA")
print("    Capacity priority: MoE + KV Cache")
print("    Quality priority: TTT + MTP + MSA")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
print("""
1. MoE/TTT/MTP/KV Cache/MSA are MiniMind's five advanced features
2. They solve different problems: efficiency/quality/speed/capacity
3. Impact on training loss: MTP > MoE > MSA > TTT > KV Cache
4. Impact on inference latency: KV Cache is largest, MSA is second
5. Can be flexibly selected and combined based on hardware and task

MiniMind achieves powerful capabilities at small model scale through these features!
""")
