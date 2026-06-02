"""
Lesson 10: Generation and Inference — Making the Model Write Word by Word
=========================================================================

How do we make a trained model "speak"?
Answer: Autoregressive generation — predict one word at a time, append the
prediction to the input, then predict the next word.

Generation process:
  Input: "Today the"
  -> Model predicts "weather" -> Input becomes "Today the weather"
  -> Model predicts "is"      -> Input becomes "Today the weather is"
  -> Model predicts "nice"    -> Input becomes "Today the weather is nice"
  -> Encounter end-of-sequence token, stop generating

Key concepts:
  - Temperature: controls randomness in generation
  - Top-K sampling: only sample from the K most probable words
  - Top-P sampling: only sample from words whose cumulative probability exceeds P
  - KV Cache: cache computed key-value pairs to avoid redundant computation

Run: python lessons_en/lesson10_generation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Reuse model components
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


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        residual = x
        x = residual + self.self_attn(self.input_layernorm(x), cos, sin)
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMindGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 num_kv_heads, head_dim, intermediate_size, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        cos = self.freqs_cos[:seq_len]
        sin = self.freqs_sin[:seq_len]
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


# ============================================================
# Step 1: Naive Generation — Greedy Decoding
# ============================================================

print("=" * 60)
print("Experiment 1: Greedy Decoding — Always Pick the Most Probable Word")
print("=" * 60)

vocab_size = 100
model = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
model.eval()

def generate_greedy(model, input_ids, max_new_tokens=20):
    """Greedy generation: always pick the most probable word"""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated

input_ids = torch.tensor([[1, 5, 10, 15]])
output = generate_greedy(model, input_ids, max_new_tokens=10)

print(f"Input:  {input_ids[0].tolist()}")
print(f"Output: {output[0].tolist()}")
print(f"\nProblem with greedy decoding: always picks the most certain word, output is monotonous and repetitive")


# ============================================================
# Step 2: Temperature Sampling — Control Randomness
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: Temperature Sampling")
print("=" * 60)

logits_raw = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])

for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
    scaled = logits_raw / temp
    probs = F.softmax(scaled, dim=-1)
    print(f"Temperature={temp:.1f}: probabilities={[f'{p:.3f}' for p in probs.tolist()]}")

print("\nLower temperature -> more concentrated probabilities -> more deterministic (conservative)")
print("Higher temperature -> more uniform probabilities -> more random (creative)")
print("Temperature=1.0 -> original probability distribution")


def generate_with_temperature(model, input_ids, max_new_tokens=20, temperature=1.0):
    """Generation with temperature"""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break
        logits = model(generated)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


input_ids = torch.tensor([[1, 5, 10]])
for temp in [0.3, 1.0, 2.0]:
    torch.manual_seed(42)
    output = generate_with_temperature(model, input_ids, max_new_tokens=8, temperature=temp)
    print(f"Temperature={temp}: {output[0].tolist()}")


# ============================================================
# Step 3: Top-K Sampling
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: Top-K Sampling — Only Sample from Top K Words")
print("=" * 60)

logits_example = torch.tensor([5.0, 3.0, 1.0, 0.5, -1.0, -3.0, -5.0, -8.0])
probs_full = F.softmax(logits_example, dim=-1)

print(f"Full probability distribution: {[f'{p:.4f}' for p in probs_full.tolist()]}")

for k in [2, 3, 5]:
    topk_vals, topk_idx = torch.topk(logits_example, k)
    filtered = torch.full_like(logits_example, float('-inf'))
    filtered[topk_idx] = topk_vals
    probs_topk = F.softmax(filtered, dim=-1)
    print(f"Top-{k}: {[f'{p:.4f}' for p in probs_topk.tolist()]}")

print("\nTop-K's role: filter out extremely low probability words, avoid generating unreasonable words")
print("K=1 is equivalent to greedy, K=vocab_size is equivalent to normal sampling")


def generate_topk(model, input_ids, max_new_tokens=20, temperature=1.0, top_k=50):
    """Top-K sampling generation"""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break
        logits = model(generated)[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, -1:]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


# ============================================================
# Step 4: Top-P (Nucleus) Sampling
# ============================================================

print("\n" + "=" * 60)
print("Experiment 4: Top-P (Nucleus) Sampling — Adaptive Filtering")
print("=" * 60)

logits_concentrated = torch.tensor([10.0, 2.0, 0.5, -1.0, -5.0])
logits_spread = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0])

for name, logits_t in [("Concentrated", logits_concentrated), ("Spread", logits_spread)]:
    probs = F.softmax(logits_t, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    print(f"\n{name}:")
    print(f"  Probabilities: {[f'{p:.4f}' for p in probs.tolist()]}")
    print(f"  Sorted:        {[f'{p:.4f}' for p in sorted_probs.tolist()]}")
    print(f"  Cumulative:    {[f'{c:.4f}' for c in cumsum.tolist()]}")

    p = 0.9
    cutoff = (cumsum - sorted_probs) < p
    n_keep = cutoff.sum().item()
    print(f"  Top-P={p}: keep first {n_keep} words")

print("\n-> Concentrated distribution needs fewer words, spread distribution needs more")
print("-> Top-P is smarter than Top-K!")


def generate_topp(model, input_ids, max_new_tokens=20, temperature=1.0, top_p=0.9):
    """Top-P (Nucleus) sampling generation"""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break
        logits = model(generated)[:, -1, :] / max(temperature, 1e-8)
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumsum - sorted_probs > top_p
        sorted_logits[sorted_mask] = float('-inf')
        logits.scatter_(1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


# ============================================================
# Step 5: Repetition Penalty
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5: Repetition Penalty — Avoid Repeating the Same Words")
print("=" * 60)

def generate_with_repetition_penalty(model, input_ids, max_new_tokens=20,
                                      temperature=1.0, top_k=50,
                                      repetition_penalty=1.2):
    """Generation with repetition penalty"""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break
        logits = model(generated)[:, -1, :] / max(temperature, 1e-8)

        if repetition_penalty > 1.0:
            for token_id in generated[0].unique():
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, -1:]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated

print("Repetition penalty mechanism:")
print("  For already-generated words, reduce their logits:")
print("    new_logits[v] = logits[v] / repetition_penalty")
print("    (repetition_penalty > 1)")
print("  Penalty > 1.0 -> already-appeared words are less likely to be selected")
print("  Common values: 1.1 ~ 1.5")


# ============================================================
# Step 6: Generation Efficiency Problem
# ============================================================

print("\n" + "=" * 60)
print("Experiment 6: Naive Generation Efficiency Problem")
print("=" * 60)

import time

model_gen = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
model_gen.eval()

input_ids = torch.tensor([[1, 5, 10, 15, 20]])

start = time.time()
output_naive = generate_greedy(model_gen, input_ids, max_new_tokens=30)
time_naive = time.time() - start

print(f"Naive generation of 30 tokens took: {time_naive*1000:.1f} ms")
print(f"Problem: Each step recomputes KV for all tokens, lots of redundant computation!")
print(f"\nOptimization: KV Cache — cache computed K and V, only compute new tokens each step")


# ============================================================
# Step 7: KV Cache Accelerated Generation
# ============================================================

print("\n" + "=" * 60)
print("Experiment 7: KV Cache — Avoid Redundant Computation")
print("=" * 60)

print("""
Naive generation (without KV Cache):
  Step 1: Input [1,5,10]     -> Compute Q,K,V -> Predict 15
  Step 2: Input [1,5,10,15]  -> Compute Q,K,V -> Predict 20  <- Recomputed KV for first 3 tokens!
  Step 3: Input [1,5,10,15,20] -> Compute Q,K,V -> Predict 25  <- Recomputed first 4!

KV Cache generation:
  Step 1: Input [1,5,10]     -> Compute K,V -> Cache -> Predict 15
  Step 2: Input [15]          -> Only compute new token's K,V -> Concatenate with cache -> Predict 20
  Step 3: Input [20]          -> Only compute new token's K,V -> Concatenate with cache -> Predict 25

-> Each step only computes 1 new token's KV, not all of them!
-> Generation speed drops from O(N^2) to O(N)
""")


# ============================================================
# Step 7.5: Complete KV Cache Implementation
# ============================================================

print("\n" + "=" * 60)
print("Experiment 7.5: Complete KV Cache Implementation (consistent with MiniMind original)")
print("=" * 60)

print("""
Core idea of KV Cache:
  1. Pre-allocate fixed-size buffers (avoid dynamic allocation overhead)
  2. Each time only compute new token's K, V, write to buffer
  3. When reading, retrieve complete K, V sequences from buffer

MiniMind's original project uses pre-allocated buffers for KVCache.
Here we implement a simplified version with the same logic.
""")


class SimpleKVCache:
    """Simplified KV Cache — consistent with MiniMind original project logic

    Pre-allocate fixed-size buffers, track valid length via len pointer.
    Each generation step only updates K, V for the new token.
    """

    def __init__(self, n_layers, bsz, max_len, n_kv_heads, head_dim, device, dtype):
        self.n_layers = n_layers
        self.max_len = max_len
        k_shape = (bsz, max_len, n_kv_heads, head_dim)
        self.k_cache = [torch.zeros(k_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(k_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.len = [0] * n_layers

    def update(self, layer_idx, new_k, new_v):
        """Write new token's K, V, return complete K, V sequences

        Args:
            layer_idx: Current layer index
            new_k: [bsz, new_seq_len, n_kv_heads, head_dim] new K
            new_v: [bsz, new_seq_len, n_kv_heads, head_dim] new V

        Returns:
            k: [bsz, total_len, n_kv_heads, head_dim] complete K sequence
            v: [bsz, total_len, n_kv_heads, head_dim] complete V sequence
        """
        cur_len = self.len[layer_idx]
        new_len = cur_len + new_k.shape[1]
        self.k_cache[layer_idx][:, cur_len:new_len].copy_(new_k)
        self.v_cache[layer_idx][:, cur_len:new_len].copy_(new_v)
        self.len[layer_idx] = new_len
        return self.k_cache[layer_idx][:, :new_len], self.v_cache[layer_idx][:, :new_len]


class AttentionWithKVCache(nn.Module):
    """GQA Attention with KV Cache"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
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

    def forward(self, x, cos, sin, kv_cache=None, layer_idx=0):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if kv_cache is not None:
            xk, xv = kv_cache.update(layer_idx, xk, xv)
        else:
            if self.n_rep > 1:
                xk = xk[:, :, :, None, :].expand(bsz, -1, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, -1, self.num_heads, self.head_dim)
                xv = xv[:, :, :, None, :].expand(bsz, -1, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, -1, self.num_heads, self.head_dim)

        if kv_cache is not None:
            total_len = xk.shape[1]
            if self.n_rep > 1:
                xk = xk[:, :, :, None, :].expand(bsz, total_len, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, total_len, self.num_heads, self.head_dim)
                xv = xv[:, :, :, None, :].expand(bsz, total_len, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, total_len, self.num_heads, self.head_dim)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)

        if kv_cache is None and seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores += mask
        elif kv_cache is not None:
            total_len = xk.shape[2]
            q_len = xq.shape[2]
            mask = torch.triu(torch.full((q_len, total_len), float('-inf'), device=x.device), diagonal=total_len - q_len + 1)
            scores += mask

        weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = weights @ xv
        output = output.transpose(1, 2).reshape(bsz, -1, self.num_heads * self.head_dim)
        return self.o_proj(output)


print("SimpleKVCache and AttentionWithKVCache defined")
print()
print("Usage:")
print("  1. First forward (Prefill): pass complete prompt, no cache")
print("  2. Subsequent steps (Decode): only pass new token, use cache for acceleration")
print()

hidden_size = 64
num_heads = 4
num_kv_heads = 2
head_dim = 16
seq_len = 8
n_layers = 2

attn_kv = AttentionWithKVCache(hidden_size, num_heads, num_kv_heads, head_dim)
freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=seq_len * 2)

x = torch.randn(1, seq_len, hidden_size)
cos_full = freqs_cos[:seq_len]
sin_full = freqs_sin[:seq_len]

out_no_cache = attn_kv(x, cos_full, sin_full, kv_cache=None)
print(f"Output without KV Cache: {out_no_cache.shape}")

kv_cache = SimpleKVCache(
    n_layers=n_layers, bsz=1, max_len=seq_len * 2,
    n_kv_heads=num_kv_heads, head_dim=head_dim,
    device=x.device, dtype=x.dtype
)

out_prefill = attn_kv(x, cos_full, sin_full, kv_cache=kv_cache, layer_idx=0)
print(f"Prefill output: {out_prefill.shape}, Cache length: {kv_cache.len[0]}")

new_token = torch.randn(1, 1, hidden_size)
cos_new = freqs_cos[seq_len:seq_len + 1]
sin_new = freqs_sin[seq_len:seq_len + 1]

out_decode = attn_kv(new_token, cos_new, sin_new, kv_cache=kv_cache, layer_idx=0)
print(f"Decode output: {out_decode.shape}, Cache length: {kv_cache.len[0]}")
print()
print("-> Prefill processes the complete prompt, Decode only processes 1 new token")
print("-> KV Cache automatically concatenates historical K, V, no redundant computation")


# ============================================================
# Step 8: Complete Generation Function
# ============================================================

print("=" * 60)
print("Experiment 8: Complete Generation Function (with all sampling strategies)")
print("=" * 60)

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0,
             top_k=0, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
    """Complete generation function

    Args:
        model: Language model
        input_ids: [1, seq_len] input token IDs
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-K sampling parameter (0=disabled)
        top_p: Top-P sampling parameter (0=disabled)
        repetition_penalty: Repetition penalty coefficient (1.0=no penalty)
        eos_token_id: End-of-sequence token ID

    Returns:
        generated: [1, seq_len + n] complete generated sequence
    """
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break

        logits = model(generated)
        next_logits = logits[:, -1, :].clone()

        if repetition_penalty > 1.0:
            for token_id in generated[0].unique():
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumsum - sorted_probs > top_p
            sorted_logits[sorted_mask] = float('-inf')
            next_logits.scatter_(1, sorted_idx, sorted_logits)

        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


input_ids = torch.tensor([[1, 5, 10, 15]])
strategies = [
    ("Greedy (temp=0)", dict(temperature=0)),
    ("Low temp (0.3)", dict(temperature=0.3)),
    ("Normal (temp=1.0)", dict(temperature=1.0)),
    ("High temp (2.0)", dict(temperature=2.0)),
    ("Top-K=5", dict(temperature=1.0, top_k=5)),
    ("Top-P=0.9", dict(temperature=1.0, top_p=0.9)),
]

for name, kwargs in strategies:
    torch.manual_seed(42)
    output = generate(model_gen, input_ids, max_new_tokens=8, **kwargs)
    print(f"{name:20s}: {output[0].tolist()}")


# ============================================================
# Deep Dive: The Essence of Generation Strategies
# ============================================================
print("\n" + "=" * 60)
print("Deep Dive: The Essence of Generation Strategies")
print("=" * 60)

print("""
[Generation = Repeatedly "Guessing the Next Word"]
---------------------------------------------------
  Input: "Today the weather"
  Model: P(next word | Today the weather)
  Probabilities: { "nice": 0.4, "good": 0.3, "cold": 0.2, "hot": 0.05, ... }

  Steps:
    1. Pick a word (based on probability)
    2. Append to input: "Today the weather nice"
    3. Predict the next word again
    4. Repeat until <eos> or max_length

  -> This is "autoregressive" generation


[Comparison of Sampling Strategies]
------------------------------------

  Greedy:
    Always pick the most probable word
    Pros: Deterministic, suitable for factual tasks
    Cons: Monotonous, prone to loops

  Temperature:
    logits /= T
    T -> 0: Approaches greedy
    T -> 1: Original distribution
    T -> inf: Completely uniform
    Analogy: Adjusting radio "clarity"
      T=0.1: Listen to the clearest station (fixed)
      T=1.0: Listen to all stations (random)

  Top-K sampling:
    Only keep the K most probable words
    Example: K=5, vocab 10000 -> only look at top 5
    Pros: Avoids picking extremely low probability weird words
    Cons: K is hard to tune, may be unnatural when distribution is uneven

  Top-P (Nucleus) sampling:
    Sample from words whose cumulative probability reaches P
    Example: P=0.9, take words until cumulative >= 0.9
    Pros: Adaptive (fewer words when peaked, more when flat)
    Cons: Occasionally selects many words

  Combined use (recommended):
    Temperature=0.7-1.0 + Top-P=0.9
    -> Both diverse and high quality


[Diagram: Sampling Strategies]
-------------------------------

  Vocab:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  Probs:  [0.4, 0.3, 0.15, 0.05, 0.04, 0.02, 0.02, 0.01, 0.005, 0.005]

  Greedy:  Pick 1 (highest)
  Top-K=3: Pick from 1, 2, 3 proportionally
  Top-P=0.85: 1+2+3 = 0.85 -> Pick from 1, 2, 3
  Top-P=0.95: Add 4 -> Pick from 1, 2, 3, 4

  After Temperature T=2.0:
    logits /= 2 -> probabilities more uniform
    P(1) ~ 0.25, P(2) ~ 0.21, ...
    -> Probability of picking 1 decreases, others increase

  After Temperature T=0.5:
    logits /= 0.5 -> probabilities more concentrated
    P(1) ~ 0.55, P(2) ~ 0.30, ...
    -> Probability of picking 1 increases


[Why KV Cache Accelerates?]
----------------------------
  Without KV Cache:
    Generate word 1: process tokens [1, 2, 3] (3 tokens)
    Generate word 2: process tokens [1, 2, 3, 1_new] (4 tokens)
    Generate word 3: process tokens [1, 2, 3, 1_new, 2_new] (5 tokens)
    ...
    Generate word n: process n+2 tokens
    Total computation: O(N^2) (N = generation length)

  With KV Cache:
    Generate word 1: process [1, 2, 3], cache K, V
    Generate word 2: only process [1_new], reuse cached K, V
    Generate word 3: only process [2_new], reuse cached K, V
    ...
    Total computation: O(N) (mainly new token)

  Speedup: O(N^2) / O(N) = N times!
  Generating 1000 words: 1000x speedup!

  Implementation:
    K, V shape: [batch, num_layers, seq_len, head_dim]
    Pre-allocate buffer for max length
    Only update the portion for new tokens each step
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Greedy vs Sampling
  Is greedy decoding output always optimal? Why?

[Exercise 2] Temperature Selection
  When generating code, should temperature be high or low?
  When generating stories, should temperature be high or low?

[Exercise 3] Top-K vs Top-P
  What are the pros and cons of Top-K=5 and Top-P=0.9?

[Exercise 4] Repetition Penalty
  How does repetition penalty work? What happens if it's too large or too small?

[Exercise 5] KV Cache Memory
  Generating 2048 tokens, batch=4, 12 layers, head_dim=64, num_kv_heads=4
  How much memory does KV Cache use? (Assume FP16, i.e., 2 bytes/element)

[Exercise 6] Generation Stopping Conditions
  When should generation stop? List 3 ways.
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Greedy is NOT always optimal! This is counterintuitive but important")
print()
print("  Counter-example:")
print("    Input: 'I want'")
print("    Probabilities: 'apple'(0.4), 'pear'(0.35), 'banana'(0.25)")
print("    Greedy picks: apple -> 'I want apple'")
print("    But 'I want pear' might actually be more natural")
print("    -> Local optimum != global optimum")
print()
print("  Essence:")
print("    Greedy only considers the current step, not what follows")
print("    Sampling preserves possibilities, occasionally finding better global solutions")
print()
print("  In practice:")
print("    Machine translation: greedy ~ BLEU 30, beam search ~ BLEU 35")
print("    Text generation: greedy is too monotonous, sampling is better")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Generating code:")
print("    - Temperature should be low (0.0 - 0.3)")
print("    - Reason: Code requires precision, high temperature produces syntax errors")
print("    - Example: def func(): must be exactly this, not random")
print()
print("  Generating stories:")
print("    - Temperature should be medium-high (0.7 - 1.2)")
print("    - Reason: Stories need creativity, too deterministic is boring")
print("    - Works better with Top-P=0.9")
print()
print("  Extreme comparison:")
print("    Temperature=0.0: 'Once upon a time, there was a mountain, in the mountain was a temple...'")
print("    Temperature=2.0: 'Once upon a time, there was a boat, in the boat was a cat, the cat was dancing...'")
print()
print("  Empirical values:")
print("    Code/Translation: T=0.0-0.2")
print("    Summarization/QA: T=0.3-0.5")
print("    Chat/Dialogue:    T=0.7-0.9")
print("    Creative writing: T=1.0-1.3")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Top-K:")
print("    Pros: Simple and intuitive, definitely excludes low probability words")
print("    Cons: K is fixed, inflexible when distribution is uneven")
print("    Counter-example: When concentrated (P(1)=0.99, others 0.01), K=5 includes words we don't want 99% of the time")
print()
print("  Top-P:")
print("    Pros: Adaptive, fewer words when peaked, more when flat")
print("    Cons: Slightly more complex implementation, occasionally selects many words")
print()
print("  Comparison:")
print("    K=5:  Always only looks at 5 words")
print("    P=0.9: When concentrated looks at 1-2, when spread looks at 100+")
print()
print("  In practice:")
print("    Top-P is almost always better than Top-K")
print("    OpenAI, Claude both default to Top-P=0.9")
print("    Can combine: Top-P=0.9 + Top-K=50 (double insurance)")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  How repetition penalty works:")
print()
print("  For already-generated words, reduce their logits:")
print("    new_logits[v] = logits[v] / repetition_penalty")
print("    (repetition_penalty > 1)")
print()
print("  Too large (e.g., 5.0):")
print("    - Model 'fears' repetition, forced to pick other words")
print("    - May produce strange synonym stacking")
print("    - Text feels unnatural")
print()
print("  Too small (e.g., 1.05):")
print("    - Penalty too light, still repeats")
print("    - Loses the purpose of penalty")
print()
print("  Empirical values:")
print("    repetition_penalty = 1.1 (recommended)")
print("    LLaMA uses 1.1, ChatGLM uses 1.2")
print()
print("  Advanced methods:")
print("    - Frequency penalty: the more occurrences, the larger the penalty")
print("    - Presence penalty: penalize if present, regardless of count")
print("    - No Repeat N-Gram: forbid repeating n consecutive words")

# Exercise 5
print("\n[Exercise 5 Answer]")
seq_len = 2048
batch = 4
num_layers = 12
head_dim = 64
num_kv_heads = 4
fp16_bytes = 2

total_elements = 2 * batch * num_layers * seq_len * num_kv_heads * head_dim
total_bytes = total_elements * fp16_bytes
total_mb = total_bytes / (1024 * 1024)
total_gb = total_mb / 1024

print(f"  Config: bs={batch}, layers={num_layers}, seq={seq_len}")
print(f"  num_kv_heads={num_kv_heads}, head_dim={head_dim}")
print(f"  K and V each: {batch} x {num_layers} x {seq_len} x {num_kv_heads} x {head_dim}")
print(f"  K elements: {batch * num_layers * seq_len * num_kv_heads * head_dim:,}")
print(f"  K+V total elements: {total_elements:,}")
print(f"  Bytes: {total_bytes:,}")
print(f"  = {total_mb:.1f} MB = {total_gb:.2f} GB")
print()
print(f"  -> Long sequences + many layers, KV Cache memory overhead is significant")
print(f"  -> This is why long-context inference needs special optimization (Flash Attn, PagedAttention, etc.)")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  1. EOS token (most common):")
print("     Stop when model generates <eos>")
print("     Training data must include <eos>")
print()
print("  2. Maximum length (max_length):")
print("     Stop when max_length tokens are generated")
print("     Prevents infinite generation")
print()
print("  3. Stop strings:")
print("     Stop when a specific string is detected")
print("     Example: Q: ... A: <eos>")
print()
print("  Other stopping conditions:")
print("    - Repetition detection: consecutive repetition N times")
print("    - Incomplete detection: sentence not finished but EOS appears")
print("    - Quality detection: perplexity too high, early termination")
print()
print("  In practice:")
print("    max_length + EOS + stop strings as triple insurance")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
print("""
1. Autoregressive generation: predict one word at a time, append to input, predict next
2. Greedy decoding: pick the most probable word, deterministic but monotonous
3. Temperature sampling: control randomness, low temp = conservative, high temp = creative
4. Top-K: only sample from the K most probable words
5. Top-P: only sample from words whose cumulative probability reaches P (smarter)
6. Repetition penalty: reduce probability of already-appeared words, avoid repetition
7. KV Cache: cache computed KV, accelerate generation

Generation strategy selection:
  Need precision (code, translation) -> Low temperature + Top-P
  Need creativity (stories, dialogue) -> Medium temperature + Top-K/P
  Need determinism (factual QA)      -> Greedy decoding

Congratulations! You've completed all the foundational MiniMind lessons!

Foundational course review:
  1. Tokenizer: Text -> Numbers
  2. Embedding: Numbers -> Vectors
  3. RMSNorm: Normalization, stable training
  4. RoPE: Rotary position encoding
  5. Attention: Words exchange information
  6. FFN: Each position independently processed
  7. Block: Assemble Attention + FFN
  8. GPT: Assemble complete model
  9. Training: Training loop
  10. Generation: Generation and inference
""")
