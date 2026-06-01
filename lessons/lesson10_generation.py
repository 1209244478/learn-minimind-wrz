"""
第10课：生成与推理 — 让模型一个字一个字地写
=============================================

训练好的模型，怎么让它"说话"？
答案：自回归生成——每次预测一个词，把预测结果加到输入中，再预测下一个词。

生成过程：
  输入: "今天"
  → 模型预测 "天气" → 输入变成 "今天天气"
  → 模型预测 "很"   → 输入变成 "今天天气很"
  → 模型预测 "好"   → 输入变成 "今天天气很好"
  → 遇到结束符，停止生成

关键概念：
  - 温度 (Temperature): 控制生成的随机性
  - Top-K 采样: 只从概率最高的 K 个词中采样
  - Top-P 采样: 只从累积概率超过 P 的词中采样
  - KV Cache: 缓存已计算的键值对，避免重复计算

运行: python lessons/lesson10_generation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 复用模型组件
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
# 第一步：最朴素的生成 — 贪心解码
# ============================================================

print("=" * 60)
print("实验1：贪心解码 — 每次选概率最高的词")
print("=" * 60)

vocab_size = 100
model = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
model.eval()

def generate_greedy(model, input_ids, max_new_tokens=20):
    """贪心生成：每次选概率最高的词"""
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

print(f"输入: {input_ids[0].tolist()}")
print(f"生成: {output[0].tolist()}")
print(f"\n贪心解码的问题：总是选最确定的词，输出很单调、重复")


# ============================================================
# 第二步：温度采样 — 控制随机性
# ============================================================

print("\n" + "=" * 60)
print("实验2：温度 (Temperature) 采样")
print("=" * 60)

logits_raw = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])

for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
    scaled = logits_raw / temp
    probs = F.softmax(scaled, dim=-1)
    print(f"温度={temp:.1f}: 概率={[f'{p:.3f}' for p in probs.tolist()]}")

print("\n温度越低 → 概率越集中 → 输出更确定（更保守）")
print("温度越高 → 概率越均匀 → 输出更随机（更有创意）")
print("温度=1.0 → 原始概率分布")


def generate_with_temperature(model, input_ids, max_new_tokens=20, temperature=1.0):
    """带温度的生成"""
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


# 不同温度的生成
input_ids = torch.tensor([[1, 5, 10]])
for temp in [0.3, 1.0, 2.0]:
    torch.manual_seed(42)
    output = generate_with_temperature(model, input_ids, max_new_tokens=8, temperature=temp)
    print(f"温度={temp}: {output[0].tolist()}")


# ============================================================
# 第三步：Top-K 采样
# ============================================================

print("\n" + "=" * 60)
print("实验3：Top-K 采样 — 只从概率最高的 K 个词中选")
print("=" * 60)

logits_example = torch.tensor([5.0, 3.0, 1.0, 0.5, -1.0, -3.0, -5.0, -8.0])
probs_full = F.softmax(logits_example, dim=-1)

print(f"完整概率分布: {[f'{p:.4f}' for p in probs_full.tolist()]}")

for k in [2, 3, 5]:
    topk_vals, topk_idx = torch.topk(logits_example, k)
    filtered = torch.full_like(logits_example, float('-inf'))
    filtered[topk_idx] = topk_vals
    probs_topk = F.softmax(filtered, dim=-1)
    print(f"Top-{k}: {[f'{p:.4f}' for p in probs_topk.tolist()]}")

print("\nTop-K 的作用：过滤掉概率极低的词，避免生成不合理的词")
print("K=1 等价于贪心，K=词表大小 等价于普通采样")


def generate_topk(model, input_ids, max_new_tokens=20, temperature=1.0, top_k=50):
    """Top-K 采样生成"""
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
# 第四步：Top-P (Nucleus) 采样
# ============================================================

print("\n" + "=" * 60)
print("实验4：Top-P (Nucleus) 采样 — 自适应过滤")
print("=" * 60)

# Top-P：选累积概率达到 P 的最少词数
# 比 Top-K 更灵活：概率集中时选少量词，概率分散时选更多词

logits_concentrated = torch.tensor([10.0, 2.0, 0.5, -1.0, -5.0])  # 集中
logits_spread = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0])           # 分散

for name, logits_t in [("集中分布", logits_concentrated), ("分散分布", logits_spread)]:
    probs = F.softmax(logits_t, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    print(f"\n{name}:")
    print(f"  概率: {[f'{p:.4f}' for p in probs.tolist()]}")
    print(f"  排序后: {[f'{p:.4f}' for p in sorted_probs.tolist()]}")
    print(f"  累积:  {[f'{c:.4f}' for c in cumsum.tolist()]}")

    p = 0.9
    cutoff = (cumsum - sorted_probs) < p
    n_keep = cutoff.sum().item()
    print(f"  Top-P={p}: 保留前 {n_keep} 个词")

print("\n→ 集中分布只需保留少量词，分散分布需要保留更多词")
print("→ Top-P 比 Top-K 更智能！")


def generate_topp(model, input_ids, max_new_tokens=20, temperature=1.0, top_p=0.9):
    """Top-P (Nucleus) 采样生成"""
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
# 第五步：重复惩罚
# ============================================================

print("\n" + "=" * 60)
print("实验5：重复惩罚 — 避免模型反复说同样的词")
print("=" * 60)

def generate_with_repetition_penalty(model, input_ids, max_new_tokens=20,
                                      temperature=1.0, top_k=50,
                                      repetition_penalty=1.2):
    """带重复惩罚的生成"""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break
        logits = model(generated)[:, -1, :] / max(temperature, 1e-8)

        # 对已生成的 token 施加惩罚
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

print("重复惩罚机制:")
print("  对已出现过的词，降低其 logits（正数除以惩罚系数，负数乘以惩罚系数）")
print("  惩罚系数 > 1.0 → 已出现的词更不容易被选中")
print("  常用值: 1.1 ~ 1.5")


# ============================================================
# 第六步：生成效率问题
# ============================================================

print("\n" + "=" * 60)
print("实验6：朴素生成的效率问题")
print("=" * 60)

import time

model_gen = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
model_gen.eval()

input_ids = torch.tensor([[1, 5, 10, 15, 20]])

# 朴素生成：每次把所有 token 重新输入模型
start = time.time()
output_naive = generate_greedy(model_gen, input_ids, max_new_tokens=30)
time_naive = time.time() - start

print(f"朴素生成 30 个 token 耗时: {time_naive*1000:.1f} ms")
print(f"问题: 每步都要重新计算所有 token 的 KV，大量重复计算！")
print(f"\n优化方案: KV Cache — 缓存已计算的 K 和 V，每步只算新 token")


# ============================================================
# 第七步：KV Cache 加速生成
# ============================================================

print("\n" + "=" * 60)
print("实验7：KV Cache — 避免重复计算")
print("=" * 60)

print("""
朴素生成（无 KV Cache）:
  步骤1: 输入 [1,5,10]     → 计算 Q,K,V → 预测 15
  步骤2: 输入 [1,5,10,15]  → 计算 Q,K,V → 预测 20  ← 重复计算了前3个token的KV！
  步骤3: 输入 [1,5,10,15,20] → 计算 Q,K,V → 预测 25  ← 重复了前4个！

KV Cache 生成:
  步骤1: 输入 [1,5,10]     → 计算 K,V → 缓存 → 预测 15
  步骤2: 输入 [15]          → 只算新token的K,V → 拼接缓存 → 预测 20
  步骤3: 输入 [20]          → 只算新token的K,V → 拼接缓存 → 预测 25

→ 每步只计算1个新token的KV，而不是全部！
→ 生成速度从 O(N^2) 降到 O(N)
""")


# ============================================================
# 第7.5步：KV Cache 的完整实现
# ============================================================

print("\n" + "=" * 60)
print("实验7.5：KV Cache 的完整实现（与 MiniMind 原始项目一致）")
print("=" * 60)

print("""
KV Cache 的核心思路：
  1. 预分配固定大小的缓冲区（避免动态分配的开销）
  2. 每次只计算新 token 的 K, V，写入缓冲区
  3. 读取时从缓冲区取出完整的 K, V 序列

MiniMind 原始项目使用预分配缓冲区的方式实现 KVCache，
这里实现一个简化版，与原始项目的逻辑完全一致。
""")


class SimpleKVCache:
    """简化版 KV Cache — 与 MiniMind 原始项目逻辑一致

    预分配固定大小的缓冲区，通过 len 指针追踪有效长度。
    每次生成步骤只更新新 token 对应的 K, V。
    """

    def __init__(self, n_layers, bsz, max_len, n_kv_heads, head_dim, device, dtype):
        self.n_layers = n_layers
        self.max_len = max_len
        k_shape = (bsz, max_len, n_kv_heads, head_dim)
        self.k_cache = [torch.zeros(k_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(k_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.len = [0] * n_layers

    def update(self, layer_idx, new_k, new_v):
        """写入新 token 的 K, V，返回完整的 K, V 序列

        Args:
            layer_idx: 当前层索引
            new_k: [bsz, new_seq_len, n_kv_heads, head_dim] 新的 K
            new_v: [bsz, new_seq_len, n_kv_heads, head_dim] 新的 V

        Returns:
            k: [bsz, total_len, n_kv_heads, head_dim] 完整的 K 序列
            v: [bsz, total_len, n_kv_heads, head_dim] 完整的 V 序列
        """
        cur_len = self.len[layer_idx]
        new_len = cur_len + new_k.shape[1]
        self.k_cache[layer_idx][:, cur_len:new_len].copy_(new_k)
        self.v_cache[layer_idx][:, cur_len:new_len].copy_(new_v)
        self.len[layer_idx] = new_len
        return self.k_cache[layer_idx][:, :new_len], self.v_cache[layer_idx][:, :new_len]


class AttentionWithKVCache(nn.Module):
    """带 KV Cache 的 GQA Attention"""

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


print("SimpleKVCache 和 AttentionWithKVCache 已定义")
print()
print("使用方式:")
print("  1. 首次前向（Prefill）: 传入完整 prompt，不使用 cache")
print("  2. 后续步骤（Decode）: 只传入新 token，使用 cache 加速")
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
print(f"无 KV Cache 输出: {out_no_cache.shape}")

kv_cache = SimpleKVCache(
    n_layers=n_layers, bsz=1, max_len=seq_len * 2,
    n_kv_heads=num_kv_heads, head_dim=head_dim,
    device=x.device, dtype=x.dtype
)

out_prefill = attn_kv(x, cos_full, sin_full, kv_cache=kv_cache, layer_idx=0)
print(f"Prefill 输出: {out_prefill.shape}, Cache 长度: {kv_cache.len[0]}")

new_token = torch.randn(1, 1, hidden_size)
cos_new = freqs_cos[seq_len:seq_len + 1]
sin_new = freqs_sin[seq_len:seq_len + 1]

out_decode = attn_kv(new_token, cos_new, sin_new, kv_cache=kv_cache, layer_idx=0)
print(f"Decode 输出: {out_decode.shape}, Cache 长度: {kv_cache.len[0]}")
print()
print("→ Prefill 处理完整 prompt，Decode 只处理 1 个新 token")
print("→ KV Cache 自动拼接历史 K, V，无需重复计算")


# ============================================================
# 第八步：完整的生成函数
# ============================================================

print("=" * 60)
print("实验8：完整的生成函数（含所有采样策略）")
print("=" * 60)

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0,
             top_k=0, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
    """完整的生成函数

    Args:
        model: 语言模型
        input_ids: [1, seq_len] 输入 token IDs
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_k: Top-K 采样参数（0=不使用）
        top_p: Top-P 采样参数（0=不使用）
        repetition_penalty: 重复惩罚系数（1.0=不惩罚）
        eos_token_id: 结束 token ID

    Returns:
        generated: [1, seq_len + n] 生成的完整序列
    """
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.max_seq_len:
            break

        # 前向传播
        logits = model(generated)
        next_logits = logits[:, -1, :].clone()

        # 重复惩罚
        if repetition_penalty > 1.0:
            for token_id in generated[0].unique():
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        # 温度缩放
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-K 过滤
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

        # Top-P 过滤
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumsum - sorted_probs > top_p
            sorted_logits[sorted_mask] = float('-inf')
            next_logits.scatter_(1, sorted_idx, sorted_logits)

        # 采样
        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        # 检查结束符
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


# 不同策略的生成对比
input_ids = torch.tensor([[1, 5, 10, 15]])
strategies = [
    ("贪心 (temp=0)", dict(temperature=0)),
    ("低温 (temp=0.3)", dict(temperature=0.3)),
    ("正常 (temp=1.0)", dict(temperature=1.0)),
    ("高温 (temp=2.0)", dict(temperature=2.0)),
    ("Top-K=5", dict(temperature=1.0, top_k=5)),
    ("Top-P=0.9", dict(temperature=1.0, top_p=0.9)),
]

for name, kwargs in strategies:
    torch.manual_seed(42)
    output = generate(model_gen, input_ids, max_new_tokens=8, **kwargs)
    print(f"{name:20s}: {output[0].tolist()}")


# ============================================================
# 深入理解：生成策略的本质
# ============================================================
print("\n" + "=" * 60)
print("深入理解：生成策略的本质")
print("=" * 60)

print("""
【生成 = 反复"猜下一个词"】
───────────────────────
  输入: "今天天气"
  模型: P(下一个词 | 今天天气)
  概率: { "好": 0.4, "不错": 0.3, "冷": 0.2, "热": 0.05, ... }

  步骤:
    1. 选一个词 (基于概率)
    2. 拼到输入: "今天天气好"
    3. 再预测下一个词
    4. 重复, 直到 <eos> 或达到 max_length

  → 这就是"自回归"生成


【采样策略的对比】
────────────────

  贪心 (Greedy):
    总是选概率最高的词
    优点: 确定性, 适合事实类
    缺点: 单调, 容易循环

  温度 (Temperature):
    logits /= T
    T → 0: 接近贪心
    T → 1: 原始分布
    T → ∞: 完全均匀
    类比: 调收音机的"清晰度"
      T=0.1: 收听最清晰的台 (固定)
      T=1.0: 收听所有台 (随机)

  Top-K 采样:
    只保留概率最高的 K 个词
    例: K=5, 词表 10000 → 只看前 5 个
    优点: 避免选到极低概率的怪词
    缺点: K 难调, 概率分布不均时可能不自然

  Top-P (Nucleus) 采样:
    从累积概率达到 P 的词中选
    例: P=0.9, 取前几个词直到累加 ≥ 0.9
    优点: 自适应 (分布尖时少选, 平缓时多选)
    缺点: 偶尔会选到很多词

  组合使用 (推荐):
    Temperature=0.7-1.0 + Top-P=0.9
    → 既有多样性, 又有质量


【图示：采样策略】
────────────────

  词表:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  概率:  [0.4, 0.3, 0.15, 0.05, 0.04, 0.02, 0.02, 0.01, 0.005, 0.005]

  贪心:  选 1 (最高)
  Top-K=3: 从 1, 2, 3 中按比例选
  Top-P=0.85: 1+2+3 = 0.85 → 从 1, 2, 3 中选
  Top-P=0.95: 加上 4 → 从 1, 2, 3, 4 中选

  温度 T=2.0 后:
    logits /= 2 → 概率更均匀
    P(1) ≈ 0.25, P(2) ≈ 0.21, ...
    → 选 1 的概率降低, 其他词概率上升

  温度 T=0.5 后:
    logits /= 0.5 → 概率更集中
    P(1) ≈ 0.55, P(2) ≈ 0.30, ...
    → 选 1 的概率升高


【KV Cache：为什么能加速？】
─────────────────────────
  无 KV Cache:
    生成第 1 个词: 处理 tokens [1, 2, 3] (3个)
    生成第 2 个词: 处理 tokens [1, 2, 3, 1_new] (4个)
    生成第 3 个词: 处理 tokens [1, 2, 3, 1_new, 2_new] (5个)
    ...
    生成第 n 个词: 处理 n+2 个 tokens
    总计算量: O(N²) (N 为生成长度)

  有 KV Cache:
    生成第 1 个词: 处理 [1, 2, 3], 缓存 K, V
    生成第 2 个词: 只处理 [1_new], 复用缓存的 K, V
    生成第 3 个词: 只处理 [2_new], 复用缓存的 K, V
    ...
    总计算量: O(N) (主要是新 token)

  加速比: O(N²) / O(N) = N 倍!
  生成 1000 个词: 1000 倍加速!

  实际实现:
    K, V 形状: [batch, num_layers, seq_len, head_dim]
    预分配最大长度的 buffer
    每次只更新新 token 对应的部分
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】贪心 vs 采样
  贪心解码输出的文本一定是最优的吗? 为什么?

【练习2】温度选择
  生成代码时, 温度应该高还是低?
  生成故事时, 温度应该高还是低?

【练习3】Top-K vs Top-P
  Top-K=5 和 Top-P=0.9 各有什么优缺点?

【练习4】重复惩罚
  重复惩罚如何工作? 太大或太小会怎样?

【练习5】KV Cache 显存
  生成 2048 个 token, batch=4, 12 层, head_dim=64, num_kv_heads=4
  KV Cache 占多少显存? (假设 FP16, 即 2 字节/元素)

【练习6】生成停止条件
  什么时候应该停止生成? 列举 3 种方式。
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  贪心不一定最优! 这是个反直觉但重要的结论")
print()
print("  反例:")
print("    输入: '我想要'")
print("    概率: '苹果'(0.4), '梨'(0.35), '香蕉'(0.25)")
print("    贪心选: 苹果 → '我想要苹果'")
print("    但 '我想要梨' 其实更通顺")
print("    → 当前最优 ≠ 全局最优")
print()
print("  本质:")
print("    贪心只考虑当前一步, 不考虑后续")
print("    采样保留可能性, 偶尔能找到更好的全局解")
print()
print("  实际:")
print("    机器翻译: 贪心 ~ BLEU 30, beam search ~ BLEU 35")
print("    文本生成: 贪心太单调, 采样更好")

# 练习2
print("\n【练习2 答案】")
print("  生成代码:")
print("    - 温度应该低 (0.0 - 0.3)")
print("    - 原因: 代码要求精确, 高温会产生语法错误")
print("    - 例: def func(): 必须这样, 不能随机")
print()
print("  生成故事:")
print("    - 温度应该中-高 (0.7 - 1.2)")
print("    - 原因: 故事需要创意, 太确定会单调")
print("    - 配合 Top-P=0.9 效果更好")
print()
print("  极端对比:")
print("    温度=0.0: '从前有座山, 山里有座庙...'")
print("    温度=2.0: '从前有座船, 船里有只猫, 猫在跳舞...'")
print()
print("  经验值:")
print("    代码/翻译: T=0.0-0.2")
print("    摘要/问答: T=0.3-0.5")
print("    聊天/对话: T=0.7-0.9")
print("    创意写作:  T=1.0-1.3")

# 练习3
print("\n【练习3 答案】")
print("  Top-K:")
print("    优点: 简单直观, 一定排除低概率词")
print("    缺点: K 固定, 分布不均时不灵活")
print("    反例: 概率集中时 (P(1)=0.99, 其他 0.01), K=5 包含 99% 都不要的词")
print()
print("  Top-P:")
print("    优点: 自适应, 概率尖时少选, 分布平时多选")
print("    缺点: 实现稍复杂, 偶尔选到很多词")
print()
print("  对比:")
print("    K=5:  始终只看 5 个")
print("    P=0.9:  概率集中时只看 1-2 个, 分布散开时看 100+ 个")
print()
print("  实际中:")
print("    Top-P 几乎总优于 Top-K")
print("    OpenAI、Claude 都默认 Top-P=0.9")
print("    可以组合: Top-P=0.9 + Top-K=50 (双保险)")

# 练习4
print("\n【练习4 答案】")
print("  重复惩罚工作原理:")
print()
print("  对已经生成过的词, 降低其 logits:")
print("    new_logits[v] = logits[v] / repetition_penalty")
print("    (repetition_penalty > 1)")
print()
print("  太大 (例如 5.0):")
print("    - 模型'怕'重复, 强行选其他词")
print("    - 可能出现奇怪的同义词堆砌")
print("    - 文本不自然")
print()
print("  太小 (例如 1.05):")
print("    - 惩罚太轻, 还是会重复")
print("    - 失去惩罚意义")
print()
print("  经验值:")
print("    repetition_penalty = 1.1 (推荐)")
print("    LLaMA 用 1.1, ChatGLM 用 1.2")
print()
print("  进阶方法:")
print("    - 频率惩罚 (frequency penalty): 出现越多, 惩罚越大")
print("    - 存在惩罚 (presence penalty): 出现就罚, 不分次数")
print("    - No Repeat N-Gram: 禁止重复 n 个连续词")

# 练习5
print("\n【练习5 答案】")
seq_len = 2048
batch = 4
num_layers = 12
head_dim = 64
num_kv_heads = 4
fp16_bytes = 2

# K 和 V 各一份
total_elements = 2 * batch * num_layers * seq_len * num_kv_heads * head_dim
total_bytes = total_elements * fp16_bytes
total_mb = total_bytes / (1024 * 1024)
total_gb = total_mb / 1024

print(f"  配置: bs={batch}, layers={num_layers}, seq={seq_len}")
print(f"  num_kv_heads={num_kv_heads}, head_dim={head_dim}")
print(f"  K, V 各 {batch} × {num_layers} × {seq_len} × {num_kv_heads} × {head_dim}")
print(f"  K 元素数: {batch * num_layers * seq_len * num_kv_heads * head_dim:,}")
print(f"  K+V 总元素: {total_elements:,}")
print(f"  字节数: {total_bytes:,}")
print(f"  = {total_mb:.1f} MB = {total_gb:.2f} GB")
print()
print(f"  → 长序列 + 多层, KV Cache 显存开销很大")
print(f"  → 这就是为什么长上下文推理需要专门优化 (Flash Attn, PagedAttention 等)")

# 练习6
print("\n【练习6 答案】")
print("  1. EOS token (最常用):")
print("     模型生成 <eos> 时停止")
print("     训练时需在数据中包含 <eos>")
print()
print("  2. 最大长度 (max_length):")
print("     生成到 max_length 个 token 时停止")
print("     防止无限生成")
print()
print("  3. 停止字符串 (stop string):")
print("     检测到特定字符串时停止")
print("     例: Q: ... A: <eos>")
print()
print("  其他停止条件:")
print("    - 重复检测: 连续重复 N 次")
print("    - 不完整检测: 句子未结束, 但 EOS 出现")
print("    - 质量检测: perplexity 太高, 提前终止")
print()
print("  实际中:")
print("    max_length + EOS + 停止字符串 三重保险")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. 自回归生成：每次预测一个词，拼接到输入中，再预测下一个
2. 贪心解码：选概率最高的词，输出确定但单调
3. 温度采样：控制随机性，低温保守，高温创意
4. Top-K：只从概率最高的 K 个词中选
5. Top-P：只从累积概率达到 P 的词中选（更智能）
6. 重复惩罚：降低已出现词的概率，避免重复
7. KV Cache：缓存已计算的 KV，加速生成

生成策略选择：
  需要精确（代码、翻译）→ 低温度 + Top-P
  需要创意（故事、对话）→ 中温度 + Top-K/P
  需要确定性（事实问答）→ 贪心解码

恭喜！你已经学完了 MiniMind 的全部基础课程！

基础课程回顾：
  1. Tokenizer: 文本 → 数字
  2. Embedding: 数字 → 向量
  3. RMSNorm: 归一化，稳定训练
  4. RoPE: 旋转位置编码
  5. Attention: 词与词交换信息
  6. FFN: 每个位置独立加工
  7. Block: 组装 Attention + FFN
  8. GPT: 组装完整模型
  9. Training: 训练循环
  10. Generation: 生成与推理

进阶课程 → lesson11_moe.py: MoE 混合专家
""")
