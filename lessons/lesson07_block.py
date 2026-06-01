"""
第7课：Transformer Block — 把 Attention + FFN 组装起来
========================================================

前几课我们学习了各个组件：
  - RMSNorm: 归一化，稳定数值
  - RoPE: 旋转位置编码，让模型知道位置
  - Attention: 词与词交换信息
  - FFN: 每个位置独立加工

现在把它们组装成一个完整的 Transformer Block！

结构（Pre-Norm 架构，MiniMind 使用）：

  输入 x
    │
    ├─→ RMSNorm → Attention → ──→ + (残差连接)
    │                              │
    │                              ├─→ RMSNorm → FFN → ──→ + (残差连接)
    │                              │                      │
    └──────────────────────────────┴──────────────────────┘
                                                          │
                                                        输出

关键设计：
  1. Pre-Norm: 先归一化再做运算（比 Post-Norm 更稳定）
  2. 残差连接: 输入直接加到输出上，防止信息丢失
  3. 两个 RMSNorm: Attention 前一个，FFN 前一个

运行: python lessons/lesson07_block.py
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


# ============================================================
# 第一步：组装 Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """一个完整的 Transformer Block（Pre-Norm 架构）"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        # Step 1: Attention 子层
        residual = x
        x_norm = self.input_layernorm(x)
        attn_out = self.self_attn(x_norm, cos, sin)
        x = residual + attn_out  # 残差连接

        # Step 2: FFN 子层
        residual = x
        x_norm = self.post_attention_layernorm(x)
        ffn_out = self.mlp(x_norm)
        x = residual + ffn_out   # 残差连接

        return x


# 实验1：Block 前向传播
print("=" * 60)
print("实验1：Transformer Block 前向传播")
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

print(f"输入形状: {x.shape}")
print(f"输出形状: {out.shape}")
print(f"Block 不改变形状，只改变内容！")

param_count = sum(p.numel() for p in block.parameters())
print(f"Block 参数量: {param_count:,}")


# ============================================================
# 第二步：残差连接的作用
# ============================================================

print("\n" + "=" * 60)
print("实验2：残差连接 — 为什么不能少？")
print("=" * 60)

class BlockNoResidual(nn.Module):
    """没有残差连接的 Block"""

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

# 堆叠多个 Block，观察信号传播
x_res = x.clone()
x_no_res = x.clone()

print("堆叠 5 个 Block 后的信号强度:")
print(f"{'层数':>4} | {'有残差 std':>12} | {'无残差 std':>12} | {'有残差 max':>12} | {'无残差 max':>12}")
print("-" * 60)

for i in range(5):
    x_res = block_with_res(x_res, cos[:8], sin[:8])
    x_no_res = block_no_res(x_no_res, cos[:8], sin[:8])
    print(f"  {i+1:>2} | {x_res.std():>12.4f} | {x_no_res.std():>12.4f} | {x_res.abs().max():>12.4f} | {x_no_res.abs().max():>12.4f}")

print("\n→ 有残差：信号稳定，不会消失或爆炸")
print("→ 无残差：信号可能快速衰减或爆炸")


# ============================================================
# 第三步：Pre-Norm vs Post-Norm
# ============================================================

print("\n" + "=" * 60)
print("实验3：Pre-Norm vs Post-Norm")
print("=" * 60)

class PostNormBlock(nn.Module):
    """Post-Norm 架构：先运算，后归一化"""

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


print("Pre-Norm (MiniMind使用):")
print("  x = x + Attention(Norm(x))")
print("  x = x + FFN(Norm(x))")
print("  归一化在运算之前 → 梯度更稳定，训练更容易")

print("\nPost-Norm (原始Transformer):")
print("  x = Norm(x + Attention(x))")
print("  x = Norm(x + FFN(x))")
print("  归一化在运算之后 → 需要精心调参，训练更难")

print("\nPre-Norm 优势:")
print("  1. 梯度可以直接通过残差路径回传（不经过 Norm）")
print("  2. 训练更稳定，不需要 warmup")
print("  3. 深层网络也能收敛")


# ============================================================
# 第四步：逐步追踪 Block 内部数据流
# ============================================================

print("\n" + "=" * 60)
print("实验4：逐步追踪 Block 内部数据流")
print("=" * 60)

block = TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
x = torch.randn(1, 4, hidden_size)

print(f"输入 x: shape={x.shape}, std={x.std():.4f}")

# Step 1: RMSNorm
residual = x
x_norm = block.input_layernorm(x)
print(f"1. RMSNorm: std={x_norm.std():.4f}")

# Step 2: Attention
attn_out = block.self_attn(x_norm, cos[:4], sin[:4])
print(f"2. Attention输出: std={attn_out.std():.4f}")

# Step 3: 残差连接
x_after_attn = residual + attn_out
print(f"3. 残差连接后: std={x_after_attn.std():.4f}")

# Step 4: RMSNorm
residual2 = x_after_attn
x_norm2 = block.post_attention_layernorm(x_after_attn)
print(f"4. RMSNorm: std={x_norm2.std():.4f}")

# Step 5: FFN
ffn_out = block.mlp(x_norm2)
print(f"5. FFN输出: std={ffn_out.std():.4f}")

# Step 6: 残差连接
x_after_ffn = residual2 + ffn_out
print(f"6. 残差连接后: std={x_after_ffn.std():.4f}")

print(f"\n完整 Block 输出: shape={x_after_ffn.shape}, std={x_after_ffn.std():.4f}")


# ============================================================
# 第五步：Block 的参数量分析
# ============================================================

print("\n" + "=" * 60)
print("实验5：Block 的参数量分布")
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

print(f"Attention 参数: {attn_params:>10,} ({attn_params/total_params*100:.1f}%)")
print(f"FFN 参数:       {ffn_params:>10,} ({ffn_params/total_params*100:.1f}%)")
print(f"RMSNorm 参数:   {norm_params:>10,} ({norm_params/total_params*100:.1f}%)")
print(f"总计:           {total_params:>10,}")
print(f"\n→ FFN 占了大部分参数！")


# ============================================================
# 第六步：堆叠多个 Block
# ============================================================

print("\n" + "=" * 60)
print("实验6：堆叠多个 Block")
print("=" * 60)

num_layers = 8
blocks = nn.ModuleList([
    TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
    for _ in range(num_layers)
])

x = torch.randn(1, 16, hidden_size)
cos = freqs_cos[:16]
sin = freqs_sin[:16]

print(f"输入: shape={x.shape}, std={x.std():.4f}")

for i, block in enumerate(blocks):
    x = block(x, cos, sin)
    if i % 2 == 0:
        print(f"Block {i+1} 输出: std={x.std():.4f}, max={x.abs().max():.4f}")

print(f"\n经过 {num_layers} 个 Block 后: shape={x.shape}")
print("信号保持稳定，这归功于 Pre-Norm + 残差连接！")

total_model_params = sum(p.numel() for p in blocks.parameters())
print(f"{num_layers} 个 Block 总参数: {total_model_params:,} = {total_model_params/1e6:.2f}M")


# ============================================================
# 深入理解：Transformer Block 的本质
# ============================================================
print("\n" + "=" * 60)
print("深入理解：Transformer Block 的本质")
print("=" * 60)

print("""
【Block 的两大功能】
──────────────────
可以把 Block 看作一个"信息处理单元":

  ┌─────────────────────────────────────────┐
  │  Block 功能:                            │
  │    1. Attention: 跨位置信息交流          │
  │    2. FFN:        单位置信息加工          │
  └─────────────────────────────────────────┘

  Attention 让词与词之间交换信息 (横向)
  FFN 让每个词独立思考 (纵向)
  残差连接保留原始信号, 避免信息丢失


【残差连接的直观解释】
────────────────────
没有残差:  y = F(x)
  → x 经过 F 之后, 原始信息可能丢失
  → 多层之后, 信号完全走样

有残差:  y = x + F(x)
  → 无论 F 学到什么, x 总会保留下来
  → 深层网络也能保留原始信号
  → 类比: 抄写时"原件"一直保存, "复印件"不断加工

数学上:
  ∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)
  → 梯度永远有 1 这个直传通道
  → 不会梯度消失!


【图示：数据流 vs 梯度流】
──────────────────────────

数据流 (前向传播):

  x ─────────────────────────────────┐
  │                                 │ ← 残差
  ▼                                 │
RMSNorm                             │
  │                                 │
  ▼                                 │
Attention                           │
  │                                 │
  ├──(+)────────────────────────────┤
  │                                 │
RMSNorm                             │
  │                                 │
  ▼                                 │
FFN                                 │
  │                                 │
  ├──(+)────────────────────────────┘
  │
  ▼
  y

梯度流 (反向传播):

  ∂L/∂y
   │
   ├──(+)──────────────→ ∂L/∂x (直传)
   │
   ▼
  ∂L/∂FFN(...)
   │
   ├──(+)──────────────→ ∂L/∂x
   │
   ▼
  ∂L/∂RMSNorm(...)
  ...


【Pre-LN vs Post-LN 的本质区别】
──────────────────────────────
Pre-LN (MiniMind 用):
  x → LN → Attn → + → LN → FFN → +
       归一化   加工  ↑   归一化   加工 ↑
                 残差             残差
  残差路径: x → x (恒等映射, 干净)
  子层输入: 总是被 LN 归一化 (范围稳定)

Post-LN (原始 Transformer):
  x → Attn → + → LN → FFN → + → LN
       加工  ↑  归一化  加工  ↑  归一化
            残差            残差
  残差路径: x → + Attn → + FFN → ... (累加)
  子层输入: 不归一化, 深层后范围爆炸

实验对比:
  Pre-LN:  训练稳定, 100+ 层也能训
  Post-LN: 训练困难, 12 层就可能梯度爆炸


【Block 之间的"分工合作"】
─────────────────────────
浅层 Block (1-3):
  - 学习局部语法 (词性、依存)
  - 注意力范围小
  - 特征抽象程度低

中层 Block (4-7):
  - 学习句法结构 (主谓宾、修饰)
  - 注意力中等范围
  - 特征开始抽象

深层 Block (8+):
  - 学习语义和推理
  - 注意力范围大
  - 高度抽象的特征

类比: 大脑皮层也分层
  - 初级视觉皮层: 边缘、颜色
  - 中级视觉皮层: 形状、纹理
  - 高级视觉皮层: 物体、场景
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】残差连接的作用
  不用残差连接, 模型会怎样?
  用 1x1 卷积 (参数小) 当残差会怎样?

【练习2】Pre-LN vs Post-LN
  为什么 Post-LN 难训练? 数学上怎么理解?

【练习3】Block 数量
  MiniMind-26M 有 8 层 Block, GPT-3 有 96 层
  为什么大模型需要更多 Block?

【练习4】Block 内参数分配
  假设 hidden=512, inter=2048, num_heads=8
  单个 Block 中, Attention 和 FFN 各占多少参数?

【练习5】信号传播
  50 层 Pre-LN + 残差, 输入 x, 输出 y
  y 包含多少层 x 的"原始信息"?

【练习6】Block 的可替换性
  同一层 Block 重复使用 12 次, 会有什么问题?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  不用残差:")
print("    - 50 层后信号几乎完全消失或爆炸")
print("    - 梯度消失, 训练失败")
print()
print("  1x1 卷积残差 (即 y = W·x + F(x), W 不恒等):")
print("    - 比不用好, 但仍有梯度问题")
print("    - W 会被训练, 可能 W 很小 (梯度消失) 或很大 (爆炸)")
print("    - 失去了'恒等映射'的优雅性质")
print()
print("  恒等残差 (y = x + F(x)) 是最简单也是最稳定的")
print("  → 残差的核心是'恒等映射', 任何参数化都会损失这个性质")

# 演示
import torch
import torch.nn as nn

# 无残差
class NoResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 恒等残差
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
print(f"  无残差输出范数: {m_no(x).norm():.4f}")
print(f"  恒等残差输出范数: {m_id(x).norm():.4f}")
print(f"  输入范数: {x.norm():.4f}")
print(f"  → 恒等残差输出与输入同量级, 无残差可能差异巨大")

# 练习2
print("\n【练习2 答案】")
print("  Post-LN 难以训练的原因:")
print()
print("  数学分析:")
print("    Post-LN 输出: y = LN(F(x) + x)")
print("    残差路径的方差随深度累积: Var(x_L) = L * Var(x_0)")
print("    → L=50 时, 方差放大 50 倍")
print("    → 子层需要处理越来越大的输入")
print("    → 权重需要不断调整范围, 训练不稳定")
print()
print("  Pre-LN 优势:")
print("    y = x + F(LN(x))")
print("    残差路径方差恒定: Var(x_L) = Var(x_0)")
print("    子层输入总是归一化后的稳定分布")
print("    → 训练稳定, 100+ 层也能训")

# 练习3
print("\n【练习3 答案】")
print("  Block 数量与模型能力的关系:")
print()
print("  类比: 思考一个问题")
print("    浅层: 第一反应 (直觉, 但粗糙)")
print("    中层: 仔细思考 (分析)")
print("    深层: 反复推敲 (推理, 反思)")
print()
print("  大模型需要更多 Block:")
print("    - 处理更复杂任务 (推理、规划)")
print("    - 记忆更多知识")
print("    - 上下文学习 (in-context learning)")
print()
print("  数量参考:")
print("    GPT-2:   12-48 层")
print("    GPT-3:   96 层 (175B 参数)")
print("    LLaMA-7B:  32 层")
print("    LLaMA-65B: 80 层")
print("    MiniMind:  8 层 (玩具级)")

# 练习4
print("\n【练习4 答案】")
hidden = 512
inter = 2048
num_heads = 8

# Attention 参数量
attn = 4 * hidden * hidden  # Q, K, V, O
print(f"  Attention: 4 × {hidden}² = {attn:,} = {attn/1e6:.2f}M")

# FFN 参数量
ffn = 3 * hidden * inter  # gate, up, down (SwiGLU)
print(f"  FFN (SwiGLU): 3 × {hidden} × {inter} = {ffn:,} = {ffn/1e6:.2f}M")

# RMSNorm 参数量 (2 个 LN)
norm = 2 * hidden
print(f"  RMSNorm (2个): 2 × {hidden} = {norm:,}")

total = attn + ffn + norm
print(f"  Block 总计: {total:,} = {total/1e6:.2f}M")
print(f"  Attention 占比: {attn/total*100:.1f}%")
print(f"  FFN 占比: {ffn/total*100:.1f}%")
print(f"  → FFN 占 Block 参数的 ~{ffn/total*100:.0f}%")

# 练习5
print("\n【练习5 答案】")
print("  数学分析 (Pre-LN + 残差):")
print()
print("    y_L = x_0 + sum_{i=0}^{L-1} F_i(LN(x_i))")
print()
print("  每一层都贡献 1 份的 F_i(LN(x_i))")
print("  所以 y_L 包含:")
print("    - 1 份的 x_0 (原始信号)")
print("    - L 份的不同变换")
print()
print("  类比: 蛋糕 + 配料")
print("    x_0 是蛋糕胚 (基础)")
print("    F_i 是每一层加的奶油/水果 (装饰)")
print("    50 层后, 蛋糕还是那个蛋糕胚, 但装饰丰富")
print()
print("  关键: x_0 永远存在, 这就是 Pre-LN 残差的好处")

# 练习6
print("\n【练习6 答案】")
print("  同一 Block 重复使用的问题:")
print()
print("  1. 表达力受限:")
print("    - 相当于 12 个相同的函数堆叠")
print("    - 等价于一个函数的 12 次复合")
print("    - 远不如 12 个不同函数学到的特征丰富")
print()
print("  2. 训练困难:")
print("    - 梯度路径相同, 容易出现对称性破缺问题")
print("    - 实际上所有层会学得一样")
print()
print("  3. 类比:")
print("    12 个人用同一台机器工作 vs 12 个人各用一台")
print("    后者效率高得多")
print()
print("  实际: 必须用不同的 Block, 权重独立")
print("  → 实际上不同 Block 自然会学到不同模式")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. Transformer Block = RMSNorm + Attention + 残差 + RMSNorm + FFN + 残差
2. 残差连接让信号直接传递，防止深层网络梯度消失
3. Pre-Norm 比 Post-Norm 更稳定，MiniMind 使用 Pre-Norm
4. FFN 占了 Block 大部分参数
5. 多个 Block 堆叠形成深层网络，信号仍能稳定传播

完整数据流：
  输入 x
    → RMSNorm → Attention → +x (残差)
    → RMSNorm → FFN → +x (残差)
    → 输出（送给下一个 Block）

下一步 → lesson08_gpt.py：把多个 Block 组装成完整的语言模型
""")
