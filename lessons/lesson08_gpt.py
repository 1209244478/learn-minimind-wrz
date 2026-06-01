"""
第8课：GPT 语言模型 — 把所有组件组装成完整模型
=================================================

前7课我们学习了每个组件，现在组装完整的 GPT 语言模型！

GPT 的完整结构：
  输入 token_ids
    → Embedding (词嵌入)
    → N × TransformerBlock
    → RMSNorm (最终归一化)
    → LM Head (线性投影到词表大小)
    → logits (每个词的概率分数)

训练目标：给定前 t 个词，预测第 t+1 个词
  P(x_{t+1} | x_1, x_2, ..., x_t)

这就是"下一个词预测"——GPT 的核心任务。

运行: python lessons/lesson08_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 复用前几课的组件
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
# 第一步：组装完整的 GPT 模型
# ============================================================

class MiniMindGPT(nn.Module):
    """完整的 GPT 语言模型"""

    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 num_kv_heads, head_dim, intermediate_size, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 1. 词嵌入
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # 2. N 个 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])

        # 3. 最终归一化
        self.norm = RMSNorm(hidden_size)

        # 4. LM Head（输出投影）
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 5. 权重共享：Embedding 和 LM Head 共享权重
        self.lm_head.weight = self.embed_tokens.weight

        # 6. 预计算 RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids):
        """前向传播

        Args:
            input_ids: [batch_size, seq_len] token ID 序列

        Returns:
            logits: [batch_size, seq_len, vocab_size] 每个位置的词表概率
        """
        bsz, seq_len = input_ids.shape

        # 1. 词嵌入
        hidden = self.embed_tokens(input_ids)

        # 2. 取对应位置的 RoPE
        cos = self.freqs_cos[:seq_len]
        sin = self.freqs_sin[:seq_len]

        # 3. 逐层通过 Transformer Block
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)

        # 4. 最终归一化
        hidden = self.norm(hidden)

        # 5. 输出投影
        logits = self.lm_head(hidden)

        return logits


# 实验1：构建并运行 GPT 模型
print("=" * 60)
print("实验1：构建 MiniMind GPT 模型")
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

print(f"词表大小: {vocab_size}")
print(f"隐藏维度: {hidden_size}")
print(f"层数: {num_layers}")
print(f"注意力头: {num_heads} Q头, {num_kv_heads} KV头")
print(f"\n输入: {input_ids.shape} (batch=2, seq_len=16)")
print(f"输出: {logits.shape} (batch=2, seq_len=16, vocab={vocab_size})")
print(f"每个位置输出 {vocab_size} 个分数，对应词表中每个词的概率")

total_params = sum(p.numel() for p in model.parameters())
unique_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,} = {total_params/1e6:.2f}M")
print(f"（权重共享后 Embedding 和 LM Head 共享 {vocab_size * hidden_size:,} 个参数）")


# ============================================================
# 第二步：从 logits 到预测
# ============================================================

print("\n" + "=" * 60)
print("实验2：从 logits 到下一个词的预测")
print("=" * 60)

# 取第一个样本，第5个位置的 logits
sample_logits = logits[0, 5]
print(f"位置5的 logits 形状: {sample_logits.shape}")
print(f"前10个 logits: {sample_logits[:10].tolist()}")

# softmax 转为概率
probs = F.softmax(sample_logits, dim=-1)
print(f"\n概率分布:")
print(f"  最大概率: {probs.max().item():.4f} (词ID={probs.argmax().item()})")
print(f"  最小概率: {probs.min().item():.6f}")
print(f"  概率之和: {probs.sum().item():.6f}")

# 取概率最高的前5个词
top5_probs, top5_ids = torch.topk(probs, 5)
print(f"\nTop-5 预测:")
for i in range(5):
    print(f"  词ID {top5_ids[i].item():>5d}: 概率 {top5_probs[i].item():.4f}")


# ============================================================
# 第三步：训练损失 — 交叉熵
# ============================================================

print("\n" + "=" * 60)
print("实验3：训练损失 — 下一个词预测的交叉熵")
print("=" * 60)

# 模拟训练数据
input_ids = torch.randint(0, vocab_size, (4, 32))
logits = model(input_ids)

# 语言模型的目标：用位置 t 的输出预测位置 t+1 的词
# 所以：logits[:, :-1, :] 预测 input_ids[:, 1:]
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

print(f"原始 logits: {logits.shape}")
print(f"移位后 logits: {shift_logits.shape} (去掉最后一个位置)")
print(f"移位后 labels: {shift_labels.shape} (去掉第一个位置)")
print(f"\n逻辑：位置0的输出 → 预测位置1的词")
print(f"      位置1的输出 → 预测位置2的词")
print(f"      ...")
print(f"      位置30的输出 → 预测位置31的词")

# 计算交叉熵损失
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1)
)
print(f"\n交叉熵损失: {loss.item():.4f}")
print(f"随机初始化模型的期望损失: {math.log(vocab_size):.4f} (= ln({vocab_size}))")
print(f"当前损失接近随机 → 模型还没学到任何东西")


# ============================================================
# 第四步：模型的完整数据流追踪
# ============================================================

print("\n" + "=" * 60)
print("实验4：完整数据流追踪")
print("=" * 60)

input_ids = torch.randint(0, vocab_size, (1, 8))
print(f"0. 输入 token_ids: {input_ids.shape}")

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
print(f"5. 最后位置预测: 词ID={predicted_id}, 概率={probs[predicted_id]:.4f}")


# ============================================================
# 第五步：模型参数量分析
# ============================================================

print("\n" + "=" * 60)
print("实验5：模型参数量分析")
print("=" * 60)

emb_params = sum(p.numel() for p in model.embed_tokens.parameters())
block_params = sum(p.numel() for p in model.layers.parameters())
norm_params = sum(p.numel() for p in model.norm.parameters())
head_params = sum(p.numel() for p in model.lm_head.parameters())

print(f"Embedding: {emb_params:>10,} ({emb_params/total_params*100:.1f}%)")
print(f"Blocks:    {block_params:>10,} ({block_params/total_params*100:.1f}%)")
print(f"FinalNorm: {norm_params:>10,} ({norm_params/total_params*100:.1f}%)")
print(f"LM Head:   {head_params:>10,} (权重共享，0额外参数)")
print(f"总计:      {total_params:>10,}")

print(f"\n→ Transformer Blocks 占了绝大部分参数")
print(f"→ 权重共享节省了 {emb_params:,} 个参数")


# ============================================================
# 第六步：不同规模的模型对比
# ============================================================

print("\n" + "=" * 60)
print("实验6：不同规模的模型对比")
print("=" * 60)

model_configs = [
    ("MiniMind-Small", 6400, 512, 8, 8, 4, 64, 1408),
    ("MiniMind-Medium", 6400, 768, 16, 12, 4, 64, 2048),
    ("LLaMA-7B", 32000, 4096, 32, 32, 32, 128, 11008),
]

for name, vs, hs, nl, nh, nkv, hd, inter in model_configs:
    p = vs * hs + nl * (3 * hs * hs + 2 * hs * inter + 2 * hs) + hs + vs * hs
    p_no_share = vs * hs + nl * (3 * hs * hs + 2 * hs * inter + 2 * hs) + hs + vs * hs
    p_share = p - vs * hs
    print(f"{name}:")
    print(f"  vocab={vs}, hidden={hs}, layers={nl}, heads={nh}/{nkv}")
    print(f"  参数量: ~{p_share/1e6:.0f}M (权重共享)")
    print()


# ============================================================
# 深入理解：GPT 架构的本质
# ============================================================
print("\n" + "=" * 60)
print("深入理解：GPT 架构的本质")
print("=" * 60)

print("""
【GPT 的核心思想：下一个词预测】
────────────────────────────
GPT (Generative Pre-trained Transformer) 的核心:
  给定前面的所有词, 预测下一个最可能的词

  例: "今天天气真" → ?
  模型预测: "好" (概率 0.6)
            "不错" (概率 0.2)
            "冷" (概率 0.1) ...

  这个简单的目标, 让模型学会:
    - 语法规则
    - 语义关系
    - 世界知识
    - 推理能力
    → 神奇的"涌现"!


【GPT 的数据流 (端到端)】
────────────────────────

  文本 "今天天气真"
       │
       ▼
  Tokenizer
       │
       ▼
  token IDs [101, 205, 312, 458]
       │
       ▼
  Embedding
       │
       ▼
  词向量 [4 × 512]
       │
       ▼
  ┌──────────────┐
  │  Block × 8   │  ← 层层抽象
  │  (循环 N 次) │
  └──────┬───────┘
         │
         ▼
  RMSNorm
         │
         ▼
  LM Head
         │
         ▼
  logits [4 × vocab_size]
         │
         ▼
  Softmax → 概率分布
         │
         ▼
  采样 → 下一个 token


【权重共享：为什么 Embedding 和 LM Head 可以共享？】
────────────────────────────────────────────────
直觉:
  Embedding 的 W_E[v] = "词 v 的语义向量"
  LM Head 的 W_LM[i] = "预测第 i 个词的特征向量"

  如果训练得好, 语义相近的词应该有相似的预测分数
  → W_E[v] 和 W_LM[v] 应该"长得像"
  → 共享权重是合理的

数学上:
  LM Head: logits = h · W_LM^T
  共享后: logits = h · W_E^T

  优势:
    - 参数量减少 50% (在 embedding 占大头的模型中)
    - 训练更稳定 (两边的梯度互相约束)
    - 隐藏层维度和 vocab 维度解耦

  注意: 不是所有模型都共享 (如 T5 不共享, GPT 共享)


【为什么是"自回归"？】
──────────────────
自回归 (Auto-regressive) = 一步一步生成

  生成 "我 爱 学习":
    步骤 1: 输入 "我"     → 预测 "爱"
    步骤 2: 输入 "我 爱"   → 预测 "学习"

  每一步都依赖上一步的输出, 形成"回归"过程

  对比:
    自回归 (GPT):  P(x_t | x_1..x_{t-1})
    自编码 (BERT):  P(x_masked | x_other) (双向)
    Prefix LM:      P(x_t | x_prefix, x_{<t}) (混合)

  GPT 选择自回归的原因:
    - 天然适合文本生成 (左到右)
    - 可以无限生成长度
    - 训练目标简单
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】为什么用下一个词预测?
  看似简单的任务, 为什么能学到如此多知识?

【练习2】权重共享的好处
  假设 vocab=64000, hidden=512
  共享权重能节省多少参数?

【练习3】为什么 LM Head 前要加 RMSNorm?
  最后一层 Block 后, 再加 RMSNorm 的作用是什么?

【练习4】损失函数值
  模型随机初始化时, 交叉熵损失大约是多少?
  为什么是这个值?

【练习5】GPT vs BERT
  GPT 是单向 (从左到右), BERT 是双向
  各自适合什么任务?

【练习6】生成时遇到 UNK
  如果生成的 token 是 UNK, 该怎么办?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  下一个词预测需要模型理解:")
print()
print("  1. 语法:")
print("    看到'主语' + '动词' → 预测'宾语'")
print("    模型学到: 词性、依存、句法")
print()
print("  2. 语义:")
print("    '猫吃' → '鱼/老鼠/食物'")
print("    模型学到: 概念关系、属性")
print()
print("  3. 世界知识:")
print("    '北京是' → '中国'的首都'")
print("    模型学到: 事实性知识")
print()
print("  4. 推理:")
print("    '下雨了, 应该' → '带伞'")
print("    模型学到: 因果关系、常识推理")
print()
print("  5. 涌现能力 (Scale):")
print("    参数足够大时, 出现'推理'、'思维链'等高级能力")
print("    这是'量变引起质变'的典型例子")
print()
print("  核心: 预测下一个词, 实际上是在预测'世界的一切规律'")

# 练习2
print("\n【练习2 答案】")
vocab = 64000
hidden = 512

embed_params = vocab * hidden
lm_head_params = vocab * hidden  # 不共享
shared_saving = lm_head_params   # 共享后节省的参数量

print(f"  Embedding: {vocab} × {hidden} = {embed_params:,}")
print(f"  LM Head:   {vocab} × {hidden} = {lm_head_params:,}")
print(f"  不共享总计: {embed_params + lm_head_params:,} = {(embed_params + lm_head_params)/1e6:.1f}M")
print(f"  共享总计:   {embed_params:,} = {embed_params/1e6:.1f}M")
print(f"  节省:       {shared_saving:,} = {shared_saving/1e6:.1f}M ({shared_saving/(embed_params+lm_head_params)*100:.0f}%)")
print(f"  → 在 embedding 占大头的模型中, 共享可节省 ~50% 参数")

# 练习3
print("\n【练习3 答案】")
print("  最后 RMSNorm 的作用:")
print()
print("  1. 数值稳定性:")
print("    - 深层 Block 后, 激活值范围可能变化")
print("    - RMSNorm 统一到合适的尺度, 便于 LM Head 处理")
print()
print("  2. 训练稳定性:")
print("    - 训练初期, Block 输出可能波动")
print("    - Norm 后更稳定, 梯度不会爆炸")
print()
print("  3. 与 Embedding 对齐:")
print("    - Embedding 输出是无界的实数")
print("    - LM Head 内部做 h·W^T, 如果 h 太大, logits 会爆炸")
print("    - RMSNorm 让 h 在合理范围")
print()
print("  没有这个 RMSNorm:")
print("    - 训练可能不稳定, loss 抖动")
print("    - 推理时 logits 数值过大, 接近 one-hot")

# 练习4
print("\n【练习4 答案】")
print("  随机初始化时, 损失 ≈ ln(vocab_size)")
print()
print("  数学推导:")
vocab_sizes = [1000, 8000, 32000, 50000, 100000]
print("  vocab_size  | ln(vocab_size) | 含义")
print("  " + "-" * 50)
for v in vocab_sizes:
    import math
    print(f"  {v:11} | {math.log(v):14.4f} | 初始损失")
print()
print("  原因:")
print("    交叉熵 = -sum(p_real * log(p_pred))")
print("    随机初始化时, p_pred = 1/vocab_size (均匀分布)")
print("    loss = -log(1/vocab_size) = log(vocab_size)")
print()
print("  训练目标: loss 不断下降, 表示模型学会预测")
print("  好的 LLM 训练后: loss ≈ 2-3 (困惑度 7-20)")

# 练习5
print("\n【练习5 答案】")
print("  GPT (单向自回归):")
print("    - 适合: 文本生成、续写、对话、翻译")
print("    - 优势: 能生成连贯的长文本")
print("    - 限制: 不能用于分类、问答 (需要微调)")
print()
print("  BERT (双向编码):")
print("    - 适合: 分类、问答、命名实体识别")
print("    - 优势: 双向理解, 上下文完整")
print("    - 限制: 不能直接生成文本")
print()
print("  实际中:")
print("    - 生成任务 → GPT 系 (GPT, LLaMA, Qwen)")
print("    - 理解任务 → BERT 系 (BERT, RoBERTa)")
print("    - 通用大模型 → 越来越多用 GPT 系 + 指令微调")

# 练习6
print("\n【练习6 答案】")
print("  解决方案:")
print()
print("  1. 屏蔽 UNK 的生成 (最常用):")
print("     在 logits 上把 UNK 的分数设为 -inf")
print("     模型永远不会选 UNK")
print()
print("  2. 改进 Tokenizer (根本解决):")
print("     BPE/SentencePiece 可以避免生成 UNK")
print("     把所有文本都切成已有 subword")
print()
print("  3. 替换为常见词:")
print("     生成 UNK 后, 用常见词替换")
print("     (质量会下降)")
print()
print("  MiniMind 的做法:")
print("    - 使用 BPE Tokenizer")
print("    - 训练数据覆盖广")
print("    - 推理时基本不出现 UNK")


# ============================================================
# 本课小结
# ============================================================
print("=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. GPT = Embedding + N*Block + Norm + LM Head
2. 训练目标：下一个词预测（用位置t的输出预测位置t+1的词）
3. 损失函数：交叉熵（随机初始化时损失 ≈ ln(vocab_size)）
4. 权重共享：Embedding 和 LM Head 共享权重，节省参数
5. Transformer Blocks 占了绝大部分参数

完整数据流：
  "你好世界" → Tokenizer → [42, 108, 35, 67]
  → Embedding → [[0.1,...], [0.8,...], ...]
  → Block×N → 上下文感知的向量
  → Norm + LM Head → logits [batch, seq, vocab]
  → softmax → 概率分布 → 预测下一个词

下一步 → lesson09_training.py：训练循环——让模型学会说话
""")
