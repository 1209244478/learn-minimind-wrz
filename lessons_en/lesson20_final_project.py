"""
Lesson 20: Final Project — Building a Complete Mini LLM from Scratch
=====================================================================

This lesson integrates all 19 previous lessons into a complete,
trainable, and deployable Mini LLM. Every component is implemented
from scratch with clear explanations.

Components covered:
  1. Tokenizer (BPE)
  2. Embedding + RoPE
  3. RMSNorm
  4. Multi-Head Attention (with GQA + KV Cache)
  5. SwiGLU FFN
  6. Transformer Block
  7. Complete GPT Model
  8. Training Loop
  9. Generation Strategies
  10. LoRA Fine-tuning
  11. Quantization

Run: python lessons_en/lesson20_final_project.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# Part 1: Tokenizer (Simplified BPE)
# ============================================================

print("=" * 60)
print("Part 1: Tokenizer (Simplified BPE)")
print("=" * 60)


class SimpleTokenizer:
    """Character-level tokenizer (simplified for demonstration)"""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build(self, text):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        return "".join([self.id_to_char[i] for i in ids if i in self.id_to_char])


sample_text = "hello world! this is a mini language model."
tokenizer = SimpleTokenizer()
tokenizer.build(sample_text)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Encoded: {tokenizer.encode('hello')}")
print(f"Decoded: {tokenizer.decode(tokenizer.encode('hello'))}")


# ============================================================
# Part 2: RMSNorm
# ============================================================

print("\n" + "=" * 60)
print("Part 2: RMSNorm")
print("=" * 60)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


rms = RMSNorm(64)
x = torch.randn(2, 10, 64)
print(f"Input:  {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
y = rms(x)
print(f"Output: {y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")


# ============================================================
# Part 3: RoPE (Rotary Position Embedding)
# ============================================================

print("\n" + "=" * 60)
print("Part 3: RoPE (Rotary Position Embedding)")
print("=" * 60)


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_r.unbind(-1)
    d = x.shape[-1] // 2
    cos = freqs_cos[:x.shape[1]].unsqueeze(0)
    sin = freqs_sin[:x.shape[1]].unsqueeze(0)
    out = torch.stack([x1 * cos[..., :d] - x2 * sin[..., :d],
                       x1 * sin[..., d:] + x2 * cos[..., d:]], dim=-1)
    return out.flatten(-2)


freqs_cos, freqs_sin = precompute_freqs_cis(64, 128)
print(f"Freqs cos shape: {freqs_cos.shape}")
print(f"Freqs sin shape: {freqs_sin.shape}")


# ============================================================
# Part 4: Multi-Head Attention (GQA + KV Cache)
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Multi-Head Attention (GQA + KV Cache)")
print("=" * 60)


class Attention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention and KV Cache"""

    def __init__(self, dim, n_heads, n_kv_heads=None, max_len=64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.k_cache = None
        self.v_cache = None

    def forward(self, x, freqs_cos=None, freqs_sin=None, start_pos=0):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cos is not None and freqs_sin is not None:
            q = apply_rotary_emb(q, freqs_cos[start_pos:start_pos+T],
                                 freqs_sin[start_pos:start_pos+T])
            k = apply_rotary_emb(k, freqs_cos[start_pos:start_pos+T],
                                 freqs_sin[start_pos:start_pos+T])

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(
                B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(
                B, self.n_heads, T, self.head_dim)

        if start_pos > 0 and self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        self.k_cache = k
        self.v_cache = v

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.full((T, k.shape[2]), float("-inf")), diagonal=1 + start_pos)
            attn = attn + mask[:T, :k.shape[2]].unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None


attn = Attention(dim=64, n_heads=4, n_kv_heads=2, max_len=64)
x = torch.randn(1, 10, 64)
out = attn(x, freqs_cos, freqs_sin)
print(f"Attention input:  {x.shape}")
print(f"Attention output: {out.shape}")


# ============================================================
# Part 5: SwiGLU FFN
# ============================================================

print("\n" + "=" * 60)
print("Part 5: SwiGLU FFN")
print("=" * 60)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


ffn = SwiGLUFFN(64, 256)
x = torch.randn(1, 10, 64)
print(f"FFN input:  {x.shape}")
print(f"FFN output: {ffn(x).shape}")


# ============================================================
# Part 6: Transformer Block
# ============================================================

print("\n" + "=" * 60)
print("Part 6: Transformer Block")
print("=" * 60)


class Block(nn.Module):
    """Transformer Block with pre-normalization"""

    def __init__(self, dim, n_heads, n_kv_heads=None, max_len=64):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads, max_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x, freqs_cos=None, freqs_sin=None, start_pos=0):
        x = x + self.attn(self.attn_norm(x), freqs_cos, freqs_sin, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


block = Block(dim=64, n_heads=4, n_kv_heads=2, max_len=64)
x = torch.randn(1, 10, 64)
print(f"Block input:  {x.shape}")
print(f"Block output: {block(x, freqs_cos, freqs_sin).shape}")


# ============================================================
# Part 7: Complete GPT Model
# ============================================================

print("\n" + "=" * 60)
print("Part 7: Complete GPT Model")
print("=" * 60)


class MiniGPT(nn.Module):
    """Mini GPT: Complete language model"""

    def __init__(self, vocab_size, dim=64, n_layers=4, n_heads=4,
                 n_kv_heads=2, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, max_len) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight

        freqs_cos, freqs_sin = precompute_freqs_cis(dim // n_heads, max_len * 2)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None, start_pos=0):
        B, T = ids.shape
        x = self.tok_emb(ids)

        for block in self.blocks:
            x = block(x, self.freqs_cos, self.freqs_sin, start_pos)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=20, temperature=1.0, top_k=None):
        self.eval()
        for block in self.blocks:
            block.attn.reset_cache()

        for i in range(max_new_tokens):
            start_pos = ids.shape[1] - 1 if i > 0 else 0
            ids_cond = ids[:, -self.max_len:]
            logits, _ = self(ids_cond, start_pos=start_pos)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

        for block in self.blocks:
            block.attn.reset_cache()

        return ids


model = MiniGPT(vocab_size=tokenizer.vocab_size, dim=64, n_layers=4,
                n_heads=4, n_kv_heads=2, max_len=64)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print(f"Model size (fp32): {total_params * 4 / 1024:.1f} KB")


# ============================================================
# Part 8: Training Loop
# ============================================================

print("\n" + "=" * 60)
print("Part 8: Training Loop")
print("=" * 60)

print("""
Training loop core steps:
  1. Forward pass: logits, loss = model(input, target)
  2. Backward pass: loss.backward()
  3. Gradient clipping: torch.nn.utils.clip_grad_norm_(params, max_norm)
  4. Optimizer step: optimizer.step()
  5. Zero gradients: optimizer.zero_grad()

Key components:
  - Loss: Cross-entropy for next-token prediction
  - Optimizer: AdamW (weight decay for regularization)
  - LR Schedule: Cosine annealing with warmup
  - Gradient Clipping: Prevents gradient explosion
""")


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def train(model, data, block_size=32, batch_size=4, steps=50, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()

    losses = []
    for step in range(steps):
        x, y = get_batch(data, block_size, batch_size)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:3d}/{steps}, Loss: {loss.item():.4f}")

    return losses


data = torch.tensor(tokenizer.encode(sample_text * 100), dtype=torch.long)
print(f"Training data: {len(data)} tokens")
print("Training...")
losses = train(model, data, block_size=32, batch_size=4, steps=50)


# ============================================================
# Part 9: Generation Strategies
# ============================================================

print("\n" + "=" * 60)
print("Part 9: Generation Strategies")
print("=" * 60)

print("""
Generation strategies:
  1. Greedy: Always pick the most likely token
  2. Temperature: Scale logits before softmax
     - T < 1: Sharper distribution (more deterministic)
     - T > 1: Flatter distribution (more random)
  3. Top-K: Only sample from top K most likely tokens
  4. Top-P (nucleus): Sample from smallest set with cumulative prob > P
  5. Repetition Penalty: Reduce probability of recently used tokens
""")

model.eval()
input_ids = torch.tensor([tokenizer.encode("hello")], dtype=torch.long)

print("\n[Greedy] (temperature=0.01):")
output = model.generate(input_ids, max_new_tokens=20, temperature=0.01)
print(f"  {tokenizer.decode(output[0].tolist())}")

print("\n[Temperature=0.8, Top-K=5]:")
output = model.generate(input_ids, max_new_tokens=20, temperature=0.8, top_k=5)
print(f"  {tokenizer.decode(output[0].tolist())}")

print("\n[Temperature=1.5, Top-K=10]:")
output = model.generate(input_ids, max_new_tokens=20, temperature=1.5, top_k=10)
print(f"  {tokenizer.decode(output[0].tolist())}")


# ============================================================
# Part 10: LoRA Fine-tuning
# ============================================================

print("\n" + "=" * 60)
print("Part 10: LoRA Fine-tuning")
print("=" * 60)


class LoRALinear(nn.Module):
    """LoRA: Low-Rank Adaptation for Linear layers"""

    def __init__(self, linear, r=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora


def apply_lora(model, r=4, alpha=8.0, target_modules=["q_proj", "v_proj"]):
    """Apply LoRA to specified modules"""
    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            lora_layer = LoRALinear(module, r=r, alpha=alpha)
            setattr(parent, child_name, lora_layer)

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable ratio:  {trainable_params/total_params:.2%}")
    return model


model_lora = MiniGPT(vocab_size=tokenizer.vocab_size, dim=64, n_layers=4,
                     n_heads=4, n_kv_heads=2, max_len=64)
print("Before LoRA:")
for p in model_lora.parameters():
    assert p.requires_grad

print("\nAfter LoRA:")
model_lora = apply_lora(model_lora, r=4, alpha=8.0)


# ============================================================
# Part 11: Quantization
# ============================================================

print("\n" + "=" * 60)
print("Part 11: Quantization")
print("=" * 60)


def quantize_model_int8(model):
    """Simple INT8 per-tensor symmetric quantization"""
    qmax = 127
    quantized = {}
    original_size = 0
    quantized_size = 0

    for name, param in model.named_parameters():
        original_size += param.numel() * 4
        abs_max = param.data.abs().max()
        if abs_max > 0:
            scale = abs_max / qmax
            param_int = torch.clamp(torch.round(param.data / scale), -128, 127).to(torch.int8)
            quantized[name] = (param_int, scale)
            quantized_size += param.numel() + 4
        else:
            quantized[name] = (param.data, 1.0)
            quantized_size += param.numel() * 4

    print(f"Original size:   {original_size / 1024:.1f} KB")
    print(f"Quantized size:  {quantized_size / 1024:.1f} KB")
    print(f"Compression:     {original_size / quantized_size:.1f}x")
    return quantized


quantized = quantize_model_int8(model)


# ============================================================
# Part 12: Complete Pipeline Summary
# ============================================================

print("\n" + "=" * 60)
print("Part 12: Complete Pipeline Summary")
print("=" * 60)

print("""
Complete MiniMind Pipeline:

1. Data Preparation
   Raw text -> Tokenizer -> Token IDs

2. Model Architecture
   Token IDs -> Embedding -> Transformer Blocks -> LM Head -> Logits

3. Training
   Logits -> Cross-Entropy Loss -> Backprop -> AdamW Update

4. Generation
   Input IDs -> Model -> Sample -> Append -> Repeat

5. Optimization
   - KV Cache: 2-5x faster inference
   - GQA: 30% less KV memory
   - LoRA: 100x fewer trainable params
   - Quantization: 4-8x smaller model

6. Deployment
   - llama.cpp: Local CPU/GPU
   - vLLM: Server-side
   - TensorRT-LLM: NVIDIA GPU

Architecture Overview:

  Input IDs
     |
  [Embedding]  (vocab_size -> dim, weight-shared with LM Head)
     |
  [Block 1]    (RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual)
  [Block 2]
  [Block 3]
  [Block 4]
     |
  [RMSNorm]
     |
  [LM Head]    (dim -> vocab_size)
     |
  Logits       (next token probabilities)
""")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Course Summary: 20 Lessons of LLM Knowledge")
print("=" * 60)

print("""
Lesson 00: Python & PyTorch Basics
Lesson 01: Tokenizers (Char / Word / BPE)
Lesson 02: Embeddings (Token + Position)
Lesson 03: RMSNorm (Efficient Normalization)
Lesson 04: RoPE (Rotary Position Embedding)
Lesson 05: Single-Head Attention
Lesson 06: Multi-Head Attention (MHA)
Lesson 07: Grouped Query Attention (GQA) & Multi-Query Attention (MQA)
Lesson 08: SwiGLU FFN
Lesson 09: Transformer Block (Assembly)
Lesson 10: GPT Architecture (Complete Model)
Lesson 11: Training Loop (Loss, Optimizer, Scheduler)
Lesson 12: Generation Strategies (Greedy, Temperature, Top-K/P)
Lesson 13: Optimizers (SGD, Adam, AdamW, Muon)
Lesson 14: Advanced Attention (Linear, ALiBi, Flash)
Lesson 15: Mamba (State Space Models)
Lesson 16: LoRA (Low-Rank Adaptation)
Lesson 17: YaRN (Length Extrapolation)
Lesson 18: mHC (Manifold-Constrained Hyper-Connections)
Lesson 19: Quantization (INT8/INT4 Deployment)
Lesson 20: Speculative Decoding (Draft-then-Verify)

Congratulations! You now understand the complete LLM pipeline,
from tokenization to deployment. Keep learning and building!
""")


# ============================================================
# Deep Understanding: The Big Picture
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: The Big Picture")
print("=" * 60)

print("""
[How All 20 Lessons Connect]
------------------------------

  Data Flow:
    Raw Text -> Tokenizer (L01) -> Token IDs
    Token IDs -> Embedding (L02) -> Vectors
    Vectors -> RoPE (L04) -> Position-aware Vectors
    Vectors -> Attention (L05-L07) -> Context-aware Vectors
    Vectors -> FFN (L08) -> Non-linear Vectors
    Vectors -> Block (L09) -> Deep Vectors
    Vectors -> GPT (L10) -> Logits
    Logits -> Loss (L11) -> Gradient -> Update

  Training:
    Data (L01) + Model (L02-L10) + Loss (L11) + Optimizer (L13)
    = Trained Model

  Inference:
    Trained Model + Generation (L12) + KV Cache (L11)
    = Text Generation

  Optimization:
    Faster Attention: Flash (L14)
    Faster Training: mHC (L18)
    Less Memory: GQA (L07), Mamba (L15)
    Less Parameters: LoRA (L16)
    Longer Context: YaRN (L17)
    Smaller Model: Quantization (L19)
    Faster Generation: Speculative Decoding (L20)


[Key Design Decisions in MiniMind]
------------------------------------

  1. Why RMSNorm instead of LayerNorm?
    - RMSNorm is simpler (no mean subtraction)
    - Same performance, slightly faster
    - Used by LLaMA, Mistral, etc.

  2. Why RoPE instead of learned position embeddings?
    - Captures relative positions naturally
    - Better length extrapolation
    - Used by most modern LLMs

  3. Why SwiGLU instead of ReLU FFN?
    - SwiGLU outperforms ReLU in practice
    - Gate mechanism provides better expressiveness
    - Used by LLaMA, PaLM, etc.

  4. Why GQA instead of MHA?
    - 30% less KV Cache memory
    - Minimal quality loss
    - Better for long sequences

  5. Why weight sharing (Embedding = LM Head)?
    - Fewer parameters
    - Works well for small models
    - Used by GPT-2, etc.

  6. Why AdamW instead of SGD?
    - Adaptive learning rates
    - Proper weight decay
    - More stable training


[Scaling Up: From MiniMind to Real LLMs]
------------------------------------------

  MiniMind (this course):
    dim=512, n_layers=8, n_heads=16, vocab=6400
    ~26M parameters

  Small LLM (GPT-2 scale):
    dim=768, n_layers=12, n_heads=12, vocab=50000
    ~124M parameters

  Medium LLM (LLaMA-7B scale):
    dim=4096, n_layers=32, n_heads=32, vocab=32000
    ~7B parameters

  Large LLM (LLaMA-70B scale):
    dim=8192, n_layers=80, n_heads=64, vocab=32000
    ~70B parameters

  Key differences:
    1. Scale: More layers, wider dimensions
    2. Data: Billions of tokens (not thousands)
    3. Training: Distributed across many GPUs
    4. Optimization: Careful hyperparameter tuning

  But the ARCHITECTURE is the same!
    All use: RMSNorm + RoPE + GQA + SwiGLU + Pre-norm
    -> You already understand the core of modern LLMs


[What's Next?]
---------------

  1. Train a real model:
    - Use the MiniMind codebase
    - Train on a larger dataset (e.g., Chinese Wikipedia)
    - Experiment with hyperparameters

  2. Study cutting-edge papers:
    - LLaMA 3, Mistral, Qwen
    - Mixture of Experts (MoE)
    - Multimodal models (Vision + Language)

  3. Build applications:
    - Chatbot / Assistant
    - RAG (Retrieval-Augmented Generation)
    - Agent systems

  4. Contribute to open source:
    - Hugging Face Transformers
    - vLLM, llama.cpp
    - MiniMind itself!

  The journey of a thousand miles begins with a single step.
  You've taken 20 steps. Keep going!
""")


# ============================================================
# Final Exercises
# ============================================================
print("\n" + "=" * 60)
print("Final Exercises")
print("=" * 60)

print("""
[Exercise 1] Architecture Design
  Design a 100M parameter model. Choose:
  - dim, n_layers, n_heads, n_kv_heads, vocab_size
  - Estimate total parameter count

[Exercise 2] Training Budget
  How much GPU time is needed to train a 100M model
  on 1B tokens? (Assume A100, 312 TFLOPS)

[Exercise 3] Memory Estimation
  How much GPU memory is needed for:
  a) Training a 7B model (FP32, Adam)
  b) Inference with a 7B model (FP16)
  c) Inference with a 7B model (INT4)

[Exercise 4] Speedup Combination
  If you apply KV Cache (3x), Quantization (2x),
  and Speculative Decoding (2.5x), what is the
  total speedup? Is it simply 3 × 2 × 2.5?

[Exercise 5] LoRA vs Full Fine-tuning
  When should you use LoRA instead of full fine-tuning?
  When is full fine-tuning better?

[Exercise 6] Deployment Decision
  You have a 7B model and need to serve it:
  a) On a phone (6GB RAM)
  b) On a laptop (16GB RAM, no GPU)
  c) On a server (8x A100)
  What quantization and deployment strategy for each?

═══════════════════════════════════════════
[Hands-On Experiments: Try It Yourself!]
═══════════════════════════════════════════

  Experiment 1: Change n_layers from 4 to 2, observe loss changes
    → Fewer layers = smaller model capacity, loss may be higher

  Experiment 2: Change dim from 512 to 256, observe generation quality
    → Half the dimensions = ~4x fewer parameters, generation may be worse

  Experiment 3: Change training epochs from 1000 to 100, observe results
    → Under-trained, model hasn't learned enough

  Experiment 4: Change learning rate from 3e-4 to 1e-2, observe training
    → Too large LR, loss may oscillate or even NaN

  Experiment 5: Change temperature from 0.8 to 0.1 and 2.0, compare outputs
    → 0.1: Near-deterministic output (repetitive, conservative)
    → 2.0: Very random (may produce gibberish)

  Experiment 6: Change n_kv_heads from 2 to 8 (=n_heads), compare KV Cache size
    → GQA → MHA, KV Cache grows but quality may improve

  Each experiment only requires changing one number and re-running!
  This is the best way to learn LLMs: try it, see the results.
""")


# ============================================================
# Final Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Final Exercise Answers")
print("=" * 60)

print("\n[Exercise 1 Answer]")
print("  100M parameter model design:")
print()
print("  Option A (wider, shallower):")
print("    dim=1024, n_layers=12, n_heads=16, n_kv_heads=4, vocab=32000")
print("    Embedding: 32000 × 1024 = 32.8M")
print("    Per block: ~8M (attn + ffn)")
print("    12 blocks: 96M")
print("    LM Head: shared with embedding")
print("    Total: ~129M (close to target)")
print()
print("  Option B (narrower, deeper):")
print("    dim=768, n_layers=24, n_heads=12, n_kv_heads=4, vocab=32000")
print("    Embedding: 32000 × 768 = 24.6M")
print("    Per block: ~4.5M")
print("    24 blocks: 108M")
print("    Total: ~133M")
print()
print("  Trade-off:")
print("    Option A: Faster inference (fewer layers)")
print("    Option B: Better quality (deeper network)")

print("\n[Exercise 2 Answer]")
print("  Training budget estimation:")
print()
print("  Model: 100M parameters")
print("  Data: 1B tokens")
print("  Compute needed: ~6 × 100M × 1B = 6 × 10^17 FLOPs")
print("  (6 FLOPs per parameter per token, forward + backward)")
print()
print("  A100: 312 TFLOPS (FP16/BF16 with tensor cores)")
print("  Hardware utilization: ~40% (realistic)")
print("  Effective: 312 × 0.4 = 124.8 TFLOPS")
print()
print("  Time = 6 × 10^17 / (124.8 × 10^12) = 4808 seconds ≈ 1.3 hours")
print()
print("  With overhead (data loading, communication, etc.):")
print("  Realistic estimate: 2-4 hours on a single A100")

print("\n[Exercise 3 Answer]")
print("  Memory estimation for 7B model:")
print()
print("  a) Training (FP32, Adam):")
print("    Weights: 7B × 4 = 28 GB")
print("    Gradients: 7B × 4 = 28 GB")
print("    Adam states: 7B × 4 × 2 = 56 GB")
print("    Activations: ~10 GB (depends on batch size)")
print("    Total: ~122 GB")
print("    -> Needs 2-4 A100 GPUs (80GB each)")
print()
print("  b) Inference (FP16):")
print("    Weights: 7B × 2 = 14 GB")
print("    KV Cache: ~2 GB (depends on sequence length)")
print("    Total: ~16 GB")
print("    -> Single RTX 4090 (24GB) works")
print()
print("  c) Inference (INT4):")
print("    Weights: 7B × 0.5 = 3.5 GB")
print("    KV Cache: ~2 GB")
print("    Total: ~5.5 GB")
print("    -> Single RTX 3060 (12GB) works")

print("\n[Exercise 4 Answer]")
print("  Speedup combination:")
print()
print("  The speedups are NOT simply multiplicative!")
print()
print("  Why? Because they optimize different bottlenecks:")
print("    - KV Cache: Reduces redundant computation")
print("    - Quantization: Reduces memory bandwidth")
print("    - Speculative: Reduces forward passes")
print()
print("  Rough estimate:")
print("    KV Cache: 3x (mostly helps with long sequences)")
print("    Quantization: 2x (helps with memory bandwidth)")
print("    Speculative: 2.5x (helps with generation steps)")
print()
print("  Combined: ~3 × 1.5 × 2 = 9x (not 15x)")
print("  The actual speedup is less than multiplicative because:")
print("    - After one optimization, the bottleneck shifts")
print("    - Other optimizations have less room to help")
print("    - Overhead from each technique adds up")
print()
print("  Realistic combined speedup: 6-10x")

print("\n[Exercise 5 Answer]")
print("  LoRA vs Full Fine-tuning:")
print()
print("  Use LoRA when:")
print("    - Limited GPU memory (< 24GB)")
print("    - Small dataset (< 100K examples)")
print("    - Need to fine-tune for multiple tasks (swap LoRA adapters)")
print("    - Quick experimentation")
print("    - Preserving the base model's general capabilities")
print()
print("  Use Full Fine-tuning when:")
print("    - Sufficient GPU memory (multi-GPU)")
print("    - Large dataset (> 1M examples)")
print("    - Significant domain shift (e.g., medical, legal)")
print("    - Need maximum performance on the target task")
print("    - The base model is far from the target domain")
print()
print("  Rule of thumb:")
print("    Start with LoRA, switch to full if performance is insufficient")

print("\n[Exercise 6 Answer]")
print("  Deployment strategy:")
print()
print("  a) Phone (6GB RAM):")
print("    -> INT4 quantization (3.5GB weights)")
print("    -> MLC-LLM or llama.cpp (mobile-optimized)")
print("    -> Short context length (512-1024)")
print("    -> Total: ~5GB (model + runtime + KV cache)")
print()
print("  b) Laptop (16GB RAM, no GPU):")
print("    -> INT4 quantization (3.5GB weights)")
print("    -> llama.cpp (CPU-optimized)")
print("    -> Medium context length (2048)")
print("    -> Total: ~6GB (model + runtime + KV cache)")
print("    -> Generation speed: ~5-10 tokens/sec")
print()
print("  c) Server (8x A100):")
print("    -> FP16 or INT8 (plenty of memory)")
print("    -> vLLM (PagedAttention, continuous batching)")
print("    -> Long context length (8192+)")
print("    -> Tensor parallelism across 8 GPUs")
print("    -> Throughput: ~1000+ tokens/sec")
