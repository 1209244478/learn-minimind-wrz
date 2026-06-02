"""
Lesson 8: GPT Language Model — Assembling All Components into a Complete Model
===============================================================================

In the previous 7 lessons we learned each component. Now let's assemble
the complete GPT language model!

GPT's complete structure:
  Input token_ids
    -> Embedding (word embedding)
    -> N x TransformerBlock
    -> RMSNorm (final normalization)
    -> LM Head (linear projection to vocabulary size)
    -> logits (probability scores for each word)

Training objective: given the first t words, predict the (t+1)-th word
  P(x_{t+1} | x_1, x_2, ..., x_t)

This is "next-token prediction" — the core task of GPT.

Run: python lessons_en/lesson08_gpt.py
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


# ============================================================
# Step 1: Assemble the Complete GPT Model
# ============================================================

class MiniMindGPT(nn.Module):
    """Complete GPT language model"""

    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 num_kv_heads, head_dim, intermediate_size, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 1. Word embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # 2. N Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])

        # 3. Final normalization
        self.norm = RMSNorm(hidden_size)

        # 4. LM Head (output projection)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 5. Weight sharing: Embedding and LM Head share weights
        self.lm_head.weight = self.embed_tokens.weight

        # 6. Precompute RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids):
        """Forward pass

        Args:
            input_ids: [batch_size, seq_len] token ID sequence

        Returns:
            logits: [batch_size, seq_len, vocab_size] vocabulary probabilities at each position
        """
        bsz, seq_len = input_ids.shape

        # 1. Word embedding
        hidden = self.embed_tokens(input_ids)

        # 2. Get RoPE for corresponding positions
        cos = self.freqs_cos[:seq_len]
        sin = self.freqs_sin[:seq_len]

        # 3. Pass through Transformer Blocks layer by layer
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)

        # 4. Final normalization
        hidden = self.norm(hidden)

        # 5. Output projection
        logits = self.lm_head(hidden)

        return logits


# Experiment 1: Build and Run the GPT Model
print("=" * 60)
print("Experiment 1: Build the MiniMind GPT Model")
print("=" * 60)

vocab_size = 6400
hidden_size = 256
num_layers = 4
num_heads = 8
num_kv_heads = 4
head_dim = 32
intermediate_size = 512

model = MiniMindGPT(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    intermediate_size=intermediate_size,
)

input_ids = torch.randint(0, vocab_size, (2, 16))
logits = model(input_ids)

print(f"Vocabulary size: {vocab_size}")
print(f"Hidden dimension: {hidden_size}")
print(f"Number of layers: {num_layers}")
print(f"Attention heads: {num_heads} Q heads, {num_kv_heads} KV heads")
print(f"\nInput: {input_ids.shape} (batch=2, seq_len=16)")
print(f"Output: {logits.shape} (batch=2, seq_len=16, vocab={vocab_size})")
print(f"Each position outputs {vocab_size} scores, corresponding to each word in the vocabulary")

total_params = sum(p.numel() for p in model.parameters())
unique_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,} = {total_params/1e6:.2f}M")
print(f"(After weight sharing, Embedding and LM Head share {vocab_size * hidden_size:,} parameters)")


# ============================================================
# Step 2: From Logits to Predictions
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: From Logits to Next-Word Prediction")
print("=" * 60)

# Take the first sample, position 5 logits
sample_logits = logits[0, 5]
print(f"Position 5 logits shape: {sample_logits.shape}")
print(f"First 10 logits: {sample_logits[:10].tolist()}")

# Softmax to convert to probabilities
probs = F.softmax(sample_logits, dim=-1)
print(f"\nProbability distribution:")
print(f"  Max probability: {probs.max().item():.4f} (word ID={probs.argmax().item()})")
print(f"  Min probability: {probs.min().item():.6f}")
print(f"  Sum of probabilities: {probs.sum().item():.6f}")

# Top-5 most probable words
top5_probs, top5_ids = torch.topk(probs, 5)
print(f"\nTop-5 predictions:")
for i in range(5):
    print(f"  Word ID {top5_ids[i].item():>5d}: probability {top5_probs[i].item():.4f}")


# ============================================================
# Step 3: Training Loss — Cross-Entropy
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: Training Loss — Cross-Entropy for Next-Token Prediction")
print("=" * 60)

# Simulate training data
input_ids = torch.randint(0, vocab_size, (4, 32))
logits = model(input_ids)

# Language model objective: use position t's output to predict position t+1's word
# So: logits[:, :-1, :] predicts input_ids[:, 1:]
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

print(f"Original logits: {logits.shape}")
print(f"Shifted logits: {shift_logits.shape} (remove last position)")
print(f"Shifted labels: {shift_labels.shape} (remove first position)")
print(f"\nLogic: position 0 output -> predict position 1 word")
print(f"       position 1 output -> predict position 2 word")
print(f"       ...")
print(f"       position 30 output -> predict position 31 word")

# Compute cross-entropy loss
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1)
)
print(f"\nCross-entropy loss: {loss.item():.4f}")
print(f"Expected loss for randomly initialized model: {math.log(vocab_size):.4f} (= ln({vocab_size}))")
print(f"Current loss is close to random -> model hasn't learned anything yet")


# ============================================================
# Step 4: Complete Data Flow Tracking
# ============================================================

print("\n" + "=" * 60)
print("Experiment 4: Complete Data Flow Tracking")
print("=" * 60)

input_ids = torch.randint(0, vocab_size, (1, 8))
print(f"0. Input token_ids: {input_ids.shape}")

hidden = model.embed_tokens(input_ids)
print(f"1. Embedding: {hidden.shape}, std={hidden.std():.4f}")

cos = model.freqs_cos[:8]
sin = model.freqs_sin[:8]

for i, layer in enumerate(model.layers):
    hidden = layer(hidden, cos, sin)
    print(f"2.{i+1}. Block {i+1}: {hidden.shape}, std={hidden.std():.4f}")

hidden = model.norm(hidden)
print(f"3. Final Norm: {hidden.shape}, std={hidden.std():.4f}")

logits = model.lm_head(hidden)
print(f"4. LM Head: {logits.shape}")

probs = F.softmax(logits[0, -1], dim=-1)
predicted_id = probs.argmax().item()
print(f"5. Last position prediction: word ID={predicted_id}, probability={probs[predicted_id]:.4f}")


# ============================================================
# Step 5: Model Parameter Count Analysis
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5: Model Parameter Count Analysis")
print("=" * 60)

emb_params = sum(p.numel() for p in model.embed_tokens.parameters())
block_params = sum(p.numel() for p in model.layers.parameters())
norm_params = sum(p.numel() for p in model.norm.parameters())
head_params = sum(p.numel() for p in model.lm_head.parameters())

print(f"Embedding: {emb_params:>10,} ({emb_params/total_params*100:.1f}%)")
print(f"Blocks:    {block_params:>10,} ({block_params/total_params*100:.1f}%)")
print(f"FinalNorm: {norm_params:>10,} ({norm_params/total_params*100:.1f}%)")
print(f"LM Head:   {head_params:>10,} (weight sharing, 0 extra parameters)")
print(f"Total:     {total_params:>10,}")

print(f"\n-> Transformer Blocks account for the vast majority of parameters")
print(f"-> Weight sharing saves {emb_params:,} parameters")


# ============================================================
# Step 6: Model Scale Comparison
# ============================================================

print("\n" + "=" * 60)
print("Experiment 6: Model Scale Comparison")
print("=" * 60)

model_configs = [
    ("MiniMind-Small", 6400, 512, 8, 8, 4, 64, 1408),
    ("MiniMind-Medium", 6400, 768, 16, 12, 4, 64, 2048),
    ("LLaMA-7B", 32000, 4096, 32, 32, 32, 128, 11008),
]

for name, vs, hs, nl, nh, nkv, hd, inter in model_configs:
    p = vs * hs + nl * (3 * hs * hs + 2 * hs * inter + 2 * hs) + hs + vs * hs
    p_share = p - vs * hs
    print(f"{name}:")
    print(f"  vocab={vs}, hidden={hs}, layers={nl}, heads={nh}/{nkv}")
    print(f"  Parameters: ~{p_share/1e6:.0f}M (with weight sharing)")
    print()


# ============================================================
# Deep Dive: The Essence of GPT Architecture
# ============================================================
print("\n" + "=" * 60)
print("Deep Dive: The Essence of GPT Architecture")
print("=" * 60)

print("""
[Core Idea of GPT: Next-Token Prediction]
------------------------------------------
GPT (Generative Pre-trained Transformer) core:
  Given all preceding words, predict the most likely next word

  Example: "The weather today is" -> ?
  Model predicts: "good" (probability 0.6)
                  "nice" (probability 0.2)
                  "cold" (probability 0.1) ...

  This simple objective lets the model learn:
    - Grammar rules
    - Semantic relationships
    - World knowledge
    - Reasoning ability
    -> Magical "emergence"!


[GPT Data Flow (End-to-End)]
----------------------------

  Text "The weather today is"
       |
       v
  Tokenizer
       |
       v
  token IDs [101, 205, 312, 458]
       |
       v
  Embedding
       |
       v
  Word vectors [4 x 512]
       |
       v
  +----------------+
  |  Block x 8     |  <- Layer-by-layer abstraction
  |  (loop N times)|
  +-------+--------+
          |
          v
  RMSNorm
          |
          v
  LM Head
          |
          v
  logits [4 x vocab_size]
          |
          v
  Softmax -> probability distribution
          |
          v
  Sampling -> next token


[Weight Sharing: Why Can Embedding and LM Head Share Weights?]
--------------------------------------------------------------
Intuition:
  Embedding's W_E[v] = "semantic vector of word v"
  LM Head's W_LM[i] = "feature vector for predicting word i"

  If well-trained, semantically similar words should have similar prediction scores
  -> W_E[v] and W_LM[v] should "look alike"
  -> Sharing weights is reasonable

Mathematically:
  LM Head: logits = h * W_LM^T
  After sharing: logits = h * W_E^T

  Advantages:
    - 50% fewer parameters (in models where embedding dominates)
    - More stable training (gradients from both sides constrain each other)
    - Decouples hidden dimension from vocab dimension

  Note: Not all models share (e.g., T5 doesn't share, GPT does)


[Why "Autoregressive"?]
-----------------------
Autoregressive = generating step by step

  Generating "I love learning":
    Step 1: Input "I"      -> predict "love"
    Step 2: Input "I love"  -> predict "learning"

  Each step depends on the previous step's output, forming a "regression" process

  Comparison:
    Autoregressive (GPT):  P(x_t | x_1..x_{t-1})
    Autoencoding (BERT):   P(x_masked | x_other) (bidirectional)
    Prefix LM:             P(x_t | x_prefix, x_{<t}) (hybrid)

  Why GPT chose autoregressive:
    - Naturally suited for text generation (left to right)
    - Can generate indefinitely long sequences
    - Simple training objective
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Why Next-Token Prediction?
  Such a seemingly simple task — why can it learn so much knowledge?

[Exercise 2] Benefits of Weight Sharing
  Given vocab=64000, hidden=512
  How many parameters does weight sharing save?

[Exercise 3] Why Add RMSNorm Before LM Head?
  What is the purpose of adding RMSNorm after the last Block?

[Exercise 4] Loss Function Value
  What is the approximate cross-entropy loss when the model is randomly initialized?
  Why this value?

[Exercise 5] GPT vs BERT
  GPT is unidirectional (left to right), BERT is bidirectional
  What tasks is each suited for?

[Exercise 6] Generating UNK
  What should we do if the generated token is UNK?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Next-token prediction requires the model to understand:")
print()
print("  1. Grammar:")
print("    Seeing 'subject' + 'verb' -> predict 'object'")
print("    Model learns: parts of speech, dependencies, syntax")
print()
print("  2. Semantics:")
print("    'Cat eats' -> 'fish/mouse/food'")
print("    Model learns: conceptual relationships, attributes")
print()
print("  3. World knowledge:")
print("    'Beijing is' -> 'the capital of China'")
print("    Model learns: factual knowledge")
print()
print("  4. Reasoning:")
print("    'It rained, you should' -> 'bring an umbrella'")
print("    Model learns: causal relationships, commonsense reasoning")
print()
print("  5. Emergent abilities (Scale):")
print("    When parameters are large enough, 'reasoning', 'chain-of-thought' etc. emerge")
print("    This is a classic example of 'quantitative change leading to qualitative change'")
print()
print("  Core: predicting the next word is actually predicting 'all the laws of the world'")

# Exercise 2
print("\n[Exercise 2 Answer]")
vocab = 64000
hidden = 512

embed_params = vocab * hidden
lm_head_params = vocab * hidden
shared_saving = lm_head_params

print(f"  Embedding: {vocab} x {hidden} = {embed_params:,}")
print(f"  LM Head:   {vocab} x {hidden} = {lm_head_params:,}")
print(f"  Without sharing: {embed_params + lm_head_params:,} = {(embed_params + lm_head_params)/1e6:.1f}M")
print(f"  With sharing:    {embed_params:,} = {embed_params/1e6:.1f}M")
print(f"  Savings:         {shared_saving:,} = {shared_saving/1e6:.1f}M ({shared_saving/(embed_params+lm_head_params)*100:.0f}%)")
print(f"  -> In models where embedding dominates, sharing can save ~50% parameters")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Purpose of the final RMSNorm:")
print()
print("  1. Numerical stability:")
print("    - After deep Blocks, activation ranges may vary")
print("    - RMSNorm unifies to an appropriate scale for LM Head processing")
print()
print("  2. Training stability:")
print("    - Early in training, Block outputs may fluctuate")
print("    - After Norm, more stable, gradients won't explode")
print()
print("  3. Alignment with Embedding:")
print("    - Embedding outputs are unbounded real numbers")
print("    - LM Head internally does h*W^T, if h is too large, logits explode")
print("    - RMSNorm keeps h in a reasonable range")
print()
print("  Without this RMSNorm:")
print("    - Training may be unstable, loss oscillates")
print("    - During inference, logits may be too large, approaching one-hot")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  When randomly initialized, loss = ln(vocab_size)")
print()
print("  Mathematical derivation:")
vocab_sizes = [1000, 8000, 32000, 50000, 100000]
print("  vocab_size  | ln(vocab_size) | Meaning")
print("  " + "-" * 50)
for v in vocab_sizes:
    print(f"  {v:11} | {math.log(v):14.4f} | Initial loss")
print()
print("  Reason:")
print("    Cross-entropy = -sum(p_real * log(p_pred))")
print("    When randomly initialized, p_pred = 1/vocab_size (uniform distribution)")
print("    loss = -log(1/vocab_size) = log(vocab_size)")
print()
print("  Training goal: loss keeps decreasing, indicating the model learns to predict")
print("  Well-trained LLM: loss = 2-3 (perplexity 7-20)")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  GPT (unidirectional autoregressive):")
print("    - Suited for: text generation, continuation, dialogue, translation")
print("    - Advantage: can generate coherent long texts")
print("    - Limitation: cannot be directly used for classification, QA (needs fine-tuning)")
print()
print("  BERT (bidirectional encoding):")
print("    - Suited for: classification, QA, named entity recognition")
print("    - Advantage: bidirectional understanding, complete context")
print("    - Limitation: cannot directly generate text")
print()
print("  In practice:")
print("    - Generation tasks -> GPT family (GPT, LLaMA, Qwen)")
print("    - Understanding tasks -> BERT family (BERT, RoBERTa)")
print("    - General-purpose LLMs -> increasingly using GPT family + instruction tuning")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Solutions:")
print()
print("  1. Block UNK generation (most common):")
print("     Set UNK's score to -inf in logits")
print("     Model will never choose UNK")
print()
print("  2. Improve Tokenizer (root cause solution):")
print("     BPE/SentencePiece can avoid generating UNK")
print("     All text is segmented into existing subwords")
print()
print("  3. Replace with common words:")
print("     After generating UNK, replace with a common word")
print("     (Quality will degrade)")
print()
print("  MiniMind's approach:")
print("    - Uses BPE Tokenizer")
print("    - Training data has broad coverage")
print("    - UNK rarely appears during inference")


# ============================================================
# Lesson Summary
# ============================================================
print("=" * 60)
print("Lesson Summary")
print("=" * 60)
print("""
1. GPT = Embedding + N*Block + Norm + LM Head
2. Training objective: next-token prediction (use position t's output to predict position t+1's word)
3. Loss function: cross-entropy (randomly initialized loss = ln(vocab_size))
4. Weight sharing: Embedding and LM Head share weights, saving parameters
5. Transformer Blocks account for the vast majority of parameters

Complete data flow:
  "Hello world" -> Tokenizer -> [42, 108, 35, 67]
  -> Embedding -> [[0.1,...], [0.8,...], ...]
  -> Block x N -> context-aware vectors
  -> Norm + LM Head -> logits [batch, seq, vocab]
  -> softmax -> probability distribution -> predict next word

Next -> lesson09_training.py: Training loop — teaching the model to speak
""")
