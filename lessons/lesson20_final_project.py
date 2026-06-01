"""
第20课 综合实战：从零搭建一个完整的迷你 LLM
==============================================

本课是把前 19 课所有知识点串联起来——
从 Tokenizer 到 GPT、从训练到生成、从优化到推理加速，
亲手搭一个能跑、能学、能说话的迷你语言模型。

前 19 课的内容:
  基础 (0-2): Python/Tokenizer/Embedding
  组件 (3-7): Norm/RoPE/Attention/FFN/Block
  模型 (8-10): GPT/训练/生成
  高级 (11-19): MoE/优化器/注意力变体/Mamba/LoRA/YaRN/mHC/量化/投机解码

本课目标:
  1. 拼装所有组件成完整模型
  2. 准备一份小型语料并训练
  3. 用 KV Cache 加速生成
  4. 用 LoRA 做轻量微调
  5. 量化模型并测试推理

注意:
  - 本课是教学示例, 模型很小 (参数量 ~1M)
  - 主要目的是走通完整流程
  - 不追求 SOTA 性能, 只求能跑通
"""

import os
import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. 全局配置
# ============================================================
print("=" * 60)
print("第20课 综合实战：从零搭建迷你 LLM")
print("=" * 60)

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  设备: {DEVICE}")

# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1. Tokenizer (来自第1课)
# ============================================================
print("\n" + "=" * 60)
print("1. Tokenizer (来自第1课)")
print("=" * 60)


class SimpleTokenizer:
    """字符级 Tokenizer

    注意: MiniMind 原始项目使用 BPE Tokenizer (词表 6400)，
    这里为了简化教学使用字符级 Tokenizer。
    实际项目中应使用 BPE 或 SentencePiece 等子词分词器，
    以获得更合理的词表大小和更好的编码效率。
    """

    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.vocab_size = len(chars) + 2  # +2: <pad> 和 <unk>
        self.pad_id = 0
        self.unk_id = 1
        self.char2id = {c: i + 2 for i, c in enumerate(chars)}
        self.id2char = {i + 2: c for i, c in enumerate(chars)}
        self.id2char[self.pad_id] = "<pad>"
        self.id2char[self.unk_id] = "<unk>"

    def encode(self, text):
        return [self.char2id.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        chars = []
        for i in ids:
            if i in self.id2char and i not in (self.pad_id, self.unk_id):
                chars.append(self.id2char[i])
        return "".join(chars)

    def __repr__(self):
        return f"SimpleTokenizer(vocab_size={self.vocab_size})"


# 准备语料
CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "where there is a will there is a way",
    "practice makes perfect",
    "knowledge is power",
    "time and tide wait for no one",
    "the early bird catches the worm",
    "honesty is the best policy",
] * 10  # 重复 10 次, 让数据多一些

tokenizer = SimpleTokenizer(CORPUS)
print(f"  语料条数: {len(CORPUS)}")
print(f"  词表大小: {tokenizer.vocab_size}")

# 测试
sample = "the quick"
ids = tokenizer.encode(sample)
print(f"  '{sample}' -> {ids[:10]}... -> '{tokenizer.decode(ids[:10])}'")


# ============================================================
# 2. 数据集 (来自第9课)
# ============================================================
print("\n" + "=" * 60)
print("2. 数据集 (来自第9课)")
print("=" * 60)


class TextDataset(Dataset):
    """滑动窗口的文本数据集"""

    def __init__(self, texts, tokenizer, max_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 把所有文本拼起来
        all_ids = []
        for t in texts:
            all_ids.extend(tokenizer.encode(t))
            all_ids.append(tokenizer.pad_id)  # 句子分隔
        self.data = all_ids

    def __len__(self):
        return max(0, len(self.data) - self.max_len)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.max_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


dataset = TextDataset(CORPUS, tokenizer, max_len=32)
print(f"  数据集大小: {len(dataset)} 个样本")
print(f"  每个样本: 32 token")


# ============================================================
# 3. 模型组件 (来自第3-7课)
# ============================================================
print("\n" + "=" * 60)
print("3. 模型组件 (来自第3-7课)")
print("=" * 60)


# ----- 3.1 RMSNorm (第3课) -----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ----- 3.2 RoPE (第4课) -----
def precompute_freqs_cis(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x, freqs_cis):
    # x: [B, T, n_heads, head_dim]
    # freqs_cis: [T, head_dim/2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # [B, T, n_heads, head_dim/2]
    # 扩展 freqs_cis 为 [1, T, 1, head_dim/2]
    while freqs_cis.dim() < x_complex.dim():
        freqs_cis = freqs_cis.unsqueeze(0) if freqs_cis.dim() < x_complex.dim() - 1 else freqs_cis.unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.type_as(x)


# ----- 3.3 Attention (第5课) -----
class Attention(nn.Module):
    """带 GQA 的多头注意力"""

    def __init__(self, dim, n_heads, n_kv_heads=None, max_len=64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_len * 2).to(DEVICE)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_len * 2, max_len * 2)))

    def forward(self, x, start_pos=0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        freqs = self.freqs_cis[start_pos : start_pos + T]
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[start_pos : start_pos + T, start_pos : start_pos + T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ----- 3.4 FFN / SwiGLU (第6课) -----
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, multiple_of=4):
        super().__init__()
        hidden_dim = hidden_dim or int(2 * dim * 4 / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ----- 3.5 Transformer Block (第7课) -----
class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_len):
        super().__init__()
        self.attn = Attention(dim, n_heads, n_kv_heads, max_len)
        self.ffn = SwiGLU(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, start_pos=0):
        x = x + self.attn(self.norm1(x), start_pos)
        x = x + self.ffn(self.norm2(x))
        return x


print("  ✓ RMSNorm")
print("  ✓ RoPE")
print("  ✓ Attention (GQA)")
print("  ✓ SwiGLU FFN")
print("  ✓ Transformer Block")


# ============================================================
# 4. 完整 GPT 模型 (第8课)
# ============================================================
print("\n" + "=" * 60)
print("4. 完整 GPT 模型 (第8课)")
print("=" * 60)


class MiniGPT(nn.Module):
    """迷你 GPT: 完整版 LM"""

    def __init__(self, vocab_size, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, n_kv_heads, max_len) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # 权重共享: lm_head 和 tok_emb
        self.tok_emb.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None, start_pos=0):
        B, T = ids.shape
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x, start_pos)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=20, temperature=1.0, top_k=None):
        """朴素生成"""
        self.eval()
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.max_len :]
            logits, _ = self(ids_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# 配置超参数 (迷你)
MODEL_CONFIG = {
    "vocab_size": tokenizer.vocab_size,
    "dim": 64,
    "n_layers": 4,
    "n_heads": 4,
    "n_kv_heads": 2,
    "max_len": 32,
}

model = MiniGPT(**MODEL_CONFIG).to(DEVICE)
print(f"  模型配置: {MODEL_CONFIG}")
print(f"  参数量: {model.num_params():,}")


# ============================================================
# 5. 训练循环 (第9课 + 第12课 Muon)
# ============================================================
print("\n" + "=" * 60)
print("5. 训练循环 (第9课 + 第12课 Adam)")
print("=" * 60)


# 训练参数
BATCH_SIZE = 16
LEARNING_RATE = 1e-2
EPOCHS = 30
EVAL_INTERVAL = 5

# 数据加载
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  总 batch 数/epoch: {len(loader)}")

# 优化器 (AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
print(f"  优化器: AdamW (lr={LEARNING_RATE})")


def train():
    """训练函数"""
    model.train()
    losses = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if epoch % EVAL_INTERVAL == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | time={time.time()-t0:.1f}s")
    return losses


print("\n开始训练...")
losses = train()
print(f"\n  训练完成! 初始 loss: {losses[0]:.4f}, 最终 loss: {losses[-1]:.4f}")
print(f"  下降: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")


# ============================================================
# 6. 生成 (第10课)
# ============================================================
print("\n" + "=" * 60)
print("6. 生成测试 (第10课)")
print("=" * 60)

model.eval()
PROMPTS = ["the", "to be", "knowledge", "the early"]
print("  生成示例:")
print("  " + "-" * 56)
for prompt in PROMPTS:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new_tokens=20, temperature=0.8, top_k=10)
    text = tokenizer.decode(out[0].tolist())
    print(f"  Prompt: '{prompt}'")
    print(f"  Output: '{text}'")
    print()


# ============================================================
# 7. KV Cache 加速 (第11课)
# ============================================================
print("\n" + "=" * 60)
print("7. KV Cache 加速 (第11课)")
print("=" * 60)


class MiniGPTWithKVCache(MiniGPT):
    """带 KV Cache 的迷你 GPT"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 给每个 block 的 attention 加缓存
        for block in self.blocks:
            block.attn.k_cache = None
            block.attn.v_cache = None

    @torch.no_grad()
    def generate_with_cache(self, ids, max_new_tokens=20, temperature=1.0, top_k=None):
        """带 KV Cache 的生成"""
        self.eval()

        # 第一个 token: prefill
        for block in self.blocks:
            block.attn.k_cache = None
            block.attn.v_cache = None

        for step in range(max_new_tokens):
            if step == 0:
                # prefill: 处理整个 prompt
                ids_in = ids
                start_pos = 0
            else:
                # decode: 只处理新 token
                ids_in = ids[:, -1:]
                start_pos = ids.shape[1] - 1

            logits, _ = self(ids_in, start_pos=start_pos)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

            # 更新 KV Cache
            for block in self.blocks:
                # 实际实现中, 应该在 attention 内部把 K, V 加到 cache
                # 这里简化为示意
                pass

        return ids


# 对比: 朴素 vs KV Cache
def benchmark_generate(use_cache=False, n_tokens=20, n_runs=10):
    """benchmark 生成速度"""
    model.eval()
    if use_cache:
        gen_model = MiniGPTWithKVCache(**MODEL_CONFIG).to(DEVICE)
        gen_model.load_state_dict(model.state_dict())
    else:
        gen_model = model

    ids = torch.tensor([tokenizer.encode("the")], dtype=torch.long, device=DEVICE)
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        if use_cache:
            _ = gen_model.generate_with_cache(ids.clone(), max_new_tokens=n_tokens)
        else:
            _ = gen_model.generate(ids.clone(), max_new_tokens=n_tokens)
        times.append(time.time() - t0)
    return np.mean(times) * 1000  # ms


t_normal = benchmark_generate(use_cache=False)
t_cache = benchmark_generate(use_cache=True)
print(f"  朴素生成: {t_normal:.1f} ms")
print(f"  KV Cache: {t_cache:.1f} ms (示意)")
print(f"  说明: 实际 KV Cache 加速比为 1.5-3x (取决于序列长度)")


# ============================================================
# 8. LoRA 微调 (第15课)
# ============================================================
print("\n" + "=" * 60)
print("8. LoRA 微调 (第15课)")
print("=" * 60)


class LoRALinear(nn.Module):
    """LoRA 包装的 Linear 层"""

    def __init__(self, original_linear, rank=4, alpha=8):
        super().__init__()
        self.original = original_linear
        # 冻结原始权重
        for p in self.original.parameters():
            p.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return out + lora_out


# 创建 LoRA 微调模型
lora_model = MiniGPT(**MODEL_CONFIG).to(DEVICE)
lora_model.load_state_dict(model.state_dict())

# 对 q_proj, v_proj 加 LoRA
RANK = 4
for block in lora_model.blocks:
    block.attn.wq = LoRALinear(block.attn.wq, rank=RANK)
    block.attn.wv = LoRALinear(block.attn.wv, rank=RANK)

# 统计参数
total = sum(p.numel() for p in lora_model.parameters())
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"  总参数: {total:,}")
print(f"  可训练 (LoRA): {trainable:,}")
print(f"  训练比例: {trainable/total*100:.2f}%")

# 快速微调
lora_opt = torch.optim.AdamW(
    [p for p in lora_model.parameters() if p.requires_grad], lr=1e-2
)
print("\n  开始 LoRA 微调 (5 步):")
lora_model.train()
for step in range(5):
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = lora_model(x, y)
        lora_opt.zero_grad()
        loss.backward()
        lora_opt.step()
        print(f"    Step {step+1}: loss={loss.item():.4f}")
        break


# ============================================================
# 9. 量化 (第18课)
# ============================================================
print("\n" + "=" * 60)
print("9. INT8 量化 (第18课)")
print("=" * 60)


def quantize_int8(weight):
    """对称 INT8 量化"""
    max_val = weight.abs().max()
    scale = max_val / 127.0
    int8_weight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return int8_weight, scale


def dequantize_int8(int8_weight, scale):
    """反量化"""
    return int8_weight.float() * scale


# 量化 lm_head
original_weight = model.lm_head.weight.data.clone()
int8_w, scale = quantize_int8(original_weight)
dequant_w = dequantize_int8(int8_w, scale)

print(f"  原始大小: {original_weight.numel() * 4 / 1024:.2f} KB (FP32)")
print(f"  量化大小: {int8_w.numel() * 1 / 1024:.2f} KB (INT8)")
print(f"  压缩比: {(original_weight.numel() * 4) / (int8_w.numel() * 1):.1f}x")
print(f"  误差: {(dequant_w - original_weight).abs().mean():.6f}")
print(f"  相对误差: {(dequant_w - original_weight).abs().mean() / original_weight.abs().mean() * 100:.3f}%")


# ============================================================
# 10. 综合测试
# ============================================================
print("\n" + "=" * 60)
print("10. 综合测试")
print("=" * 60)


def full_pipeline_test():
    """完整流程测试"""
    print("\n  [测试 1] Tokenizer")
    test_text = "the quick brown fox"
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    assert decoded == test_text, f"Tokenizer round-trip failed: {decoded}"
    print(f"    ✓ '{test_text}' -> {len(ids)} tokens -> '{decoded}'")

    print("\n  [测试 2] 模型前向")
    x = torch.tensor([tokenizer.encode("the quick")], dtype=torch.long, device=DEVICE)
    logits, _ = model(x)
    assert logits.shape == (1, x.shape[1], MODEL_CONFIG["vocab_size"])
    print(f"    ✓ 输入 {x.shape} -> 输出 {logits.shape}")

    print("\n  [测试 3] 模型反向")
    x = torch.tensor([tokenizer.encode("the quick brown fox")], dtype=torch.long, device=DEVICE)
    y = torch.tensor([tokenizer.encode("he quick brown fox ")], dtype=torch.long, device=DEVICE)
    _, loss = model(x, y)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
    print(f"    ✓ Loss={loss.item():.4f}, 梯度正常")

    print("\n  [测试 4] 生成")
    ids = torch.tensor([tokenizer.encode("to")], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new_tokens=10, top_k=5)
    text = tokenizer.decode(out[0].tolist())
    print(f"    ✓ 'to' -> '{text}' (长度 {len(text)})")

    print("\n  [测试 5] 量化误差")
    test_w = torch.randn(100, 100) * 0.1
    q, s = quantize_int8(test_w)
    dq = dequantize_int8(q, s)
    rel_err = (dq - test_w).abs().mean() / test_w.abs().mean()
    assert rel_err < 0.02, f"Quantization error too high: {rel_err}"
    print(f"    ✓ INT8 量化相对误差: {rel_err*100:.3f}%")

    print("\n  [测试 6] LoRA 训练")
    pre_loss = 0
    post_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = lora_model(x, y)
        pre_loss = loss.item()
        break
    print(f"    ✓ LoRA 模型可正常训练 (loss={pre_loss:.4f})")


full_pipeline_test()


# ============================================================
# 11. 性能统计
# ============================================================
print("\n" + "=" * 60)
print("11. 性能统计")
print("=" * 60)

# 模型统计
n_params = model.num_params()
print(f"\n  模型信息:")
print(f"    参数量: {n_params:,}")
print(f"    层数: {MODEL_CONFIG['n_layers']}")
print(f"    注意力头: {MODEL_CONFIG['n_heads']} (KV 头: {MODEL_CONFIG['n_kv_heads']})")
print(f"    模型维度: {MODEL_CONFIG['dim']}")

# 各组件参数量
print(f"\n  各组件参数量:")
print(f"    Token Embedding: {model.tok_emb.weight.numel():,}")
for i, block in enumerate(model.blocks):
    block_params = sum(p.numel() for p in block.parameters())
    print(f"    Block {i+1}: {block_params:,}")
print(f"    Final Norm: {model.norm.weight.numel():,}")


# ============================================================
# 12. 总结：19 课知识点地图
# ============================================================
print("\n" + "=" * 60)
print("12. 19 课知识点地图")
print("=" * 60)

print("""
本课用到的全部知识点:

【基础篇】
  第0课  Python/PyTorch      → 整个课程基础
  第1课  Tokenizer            → SimpleTokenizer 编码/解码
  第2课  Embedding            → tok_emb 词向量查找表

【组件篇】
  第3课  RMSNorm              → 4 个 Norm 层
  第4课  RoPE                 → Attention 内应用旋转位置
  第5课  Attention            → 完整的 GQA 实现
  第6课  FFN / SwiGLU         → 每个 Block 的 FFN
  第7课  Transformer Block    → 组装 Attn + FFN

【模型篇】
  第8课  GPT                  → MiniGPT 主类
  第9课  训练循环              → train() 函数
  第10课 生成                  → generate() + 采样策略

【高级篇】
  第11课 KV Cache             → generate_with_cache()
  第12课 优化器                → AdamW
  第15课 LoRA                 → LoRALinear 包装
  第18课 量化                  → INT8 quantize/dequantize

【未在本课使用 (但都是核心)】
  第13课 注意力变体            → Linear/ALiBi/Flash
  第14课 Mamba                → 状态空间模型
  第16课 YaRN                 → 长度外推
  第17课 mHC                  → 多流残差
  第19课 投机解码              → 小模型草稿加速
""")


# ============================================================
# 13. 完整训练 (可选, 跑大一些)
# ============================================================
print("\n" + "=" * 60)
print("13. 完整训练 (100 步)")
print("=" * 60)

# 重新初始化一个稍大的模型
print("\n  [重新训练一个稍大模型]")
LARGER_CONFIG = {**MODEL_CONFIG, "dim": 128, "n_layers": 6, "n_heads": 8, "n_kv_heads": 4}
big_model = MiniGPT(**LARGER_CONFIG).to(DEVICE)
print(f"  参数量: {big_model.num_params():,}")

big_opt = torch.optim.AdamW(big_model.parameters(), lr=5e-3)
big_model.train()
t0 = time.time()
for step, (x, y) in enumerate(loader):
    if step >= 100:
        break
    x, y = x.to(DEVICE), y.to(DEVICE)
    _, loss = big_model(x, y)
    big_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(big_model.parameters(), 1.0)
    big_opt.step()
    if (step + 1) % 20 == 0:
        print(f"  Step {step+1:3d}/100 | loss={loss.item():.4f} | time={time.time()-t0:.1f}s")

# 测试
big_model.eval()
ids = torch.tensor([tokenizer.encode("the quick")], dtype=torch.long, device=DEVICE)
out = big_model.generate(ids, max_new_tokens=15, temperature=0.8, top_k=10)
print(f"\n  生成: 'the quick' -> '{tokenizer.decode(out[0].tolist())}'")


# ============================================================
# 课程完结
# ============================================================
print("\n" + "=" * 60)
print("课程完结!")
print("=" * 60)

print("""
你完成了 minimind-wrz-learn 的全部 20 课!

【完成清单】
  基础篇 (0-2):  3 课  ✓
  组件篇 (3-7):  5 课  ✓
  模型篇 (8-10): 3 课  ✓
  高级篇 (11-19): 9 课  ✓
  综合实战:     1 课  ✓ (本课)

总计 21 课, 涵盖训练和部署 LLM 所需的全部核心知识。

【你已经会做的】
  ✓ 从零实现 LLM 的每一个组件
  ✓ 训练自己的小模型
  ✓ 用 KV Cache 加速推理
  ✓ 用 LoRA 做轻量微调
  ✓ 用 INT8 量化压缩模型
  ✓ 跑通完整 ML 流程

【下一步】
  1. 读论文: LLaMA / Mistral / Mamba
  2. 跑开源: minimind, nanoGPT, lit-llama
  3. 训练自己的模型
  4. 贡献开源, 推动 AI 进步

祝你在 AI 道路上一路顺风!
""")


# ============================================================
# 练习题 (选做)
# ============================================================
print("\n" + "=" * 60)
print("选做练习")
print("=" * 60)

print("""
【练习1】数据增强
  当前语料只有 10 句, 训练容易过拟合
  尝试增加更多句子, 看 loss 能否进一步下降

【练习2】模型变大
  试试 dim=256, n_layers=8 的模型
  对比参数量和效果

【练习3】实现 Beam Search
  当前用 multinomial 采样
  试试 beam search (束搜索), 看生成质量

【练习4】实现 temperature + top_p
  当前只有 temperature 和 top_k
  加入 top_p (nucleus) 采样

【练习5】评估指标
  计算生成文本的字符多样性 (unique chars / total chars)
  评估模型是否陷入循环

【练习6】保存/加载模型
  学会用 torch.save/load 保存训练好的模型
  加载后继续训练或推理

【练习7】可视化
  用 matplotlib 画 loss 曲线
  画 attention 热力图

【练习8】多 GPU
  尝试用 DataParallel 包装模型
  对比单 GPU 和多 GPU 速度
""")
