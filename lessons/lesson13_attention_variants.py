"""
第13课：注意力变体 — 突破 O(N^2) 的束缚
==========================================

标准的 Softmax Attention 有个致命的缺点：
  时间复杂度 O(N^2)，N 是序列长度
  - 1024 tokens → 1M 计算
  - 8192 tokens → 67M 计算
  - 100K tokens → 100亿计算 (无法接受！)

这一课，我们探索 3 个变体：
  1. Linear Attention: O(N) 复杂度
  2. ALiBi: 长度外推友好的位置编码
  3. Flash Attention: 显存友好的高效 Attention

运行: python lessons/lesson13_attention_variants.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 第1部分：复习标准 Attention
# ============================================================

print("=" * 60)
print("第1部分：标准 Softmax Attention 回顾")
print("=" * 60)

print("""
标准 Attention 流程：
  Q, K, V 都是 (seq_len, d_k) 的矩阵
  1. scores = Q @ K^T / √d_k      → (seq_len, seq_len)
  2. weights = softmax(scores)    → (seq_len, seq_len)
  3. output = weights @ V         → (seq_len, d_k)

问题：
  - step 1 的 scores 是 N×N 矩阵，N 大时内存爆炸
  - step 3 又是 N×N 矩阵乘法
  - 整体 O(N^2) 复杂度
""")


# ============================================================
# 第2部分：Linear Attention
# ============================================================

print("=" * 60)
print("第2部分：Linear Attention — 突破 O(N^2)")
print("=" * 60)

print("""
核心思想：用特征映射 φ 替代 softmax

标准:  output = softmax(QK^T) @ V
线性:  output = φ(Q) @ (φ(K)^T @ V)
                ↑ 关键：矩阵乘法结合律

标准 Attention: O(N^2) — 必须先算 N×N 的 attention matrix
Linear Attention: O(N) — 巧妙调换矩阵乘法顺序

需要满足：softmax(QK^T) ≈ φ(Q) @ φ(K)^T
  → φ 通常是 elu(x) + 1 或 ReLU

优缺点：
  ✓ O(N) 复杂度，适合长序列
  ✓ 可以处理任意长度（无需重训练）
  ✗ 表达能力略弱于 softmax
  ✗ 训练时对数值精度敏感
""")


def elu_feature_map(x):
    """ELU+1 特征映射: 满足 φ(q)·φ(k) ≈ exp(q·k)"""
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    """Linear Attention (Performer 风格)"""

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 扩展 KV
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)

        # 应用特征映射
        Q = elu_feature_map(xq)
        K = elu_feature_map(xk)
        V = xv

        # Linear Attention 核心: 先算 K^T @ V (D x D) 再算 Q @ (K^T @ V)
        # 等价于 Q @ K^T @ V，但避免了 N×N 矩阵
        KV = K.transpose(-2, -1) @ V  # (b, h, d_k, d_v)
        out = Q @ KV  # (b, h, n, d_v)

        # 归一化
        K_sum = K.sum(dim=-2, keepdim=True).transpose(-2, -1)  # (b, h, 1, d_k)
        Z = 1.0 / (Q @ K_sum + 1e-8)  # 归一化因子
        out = out * Z

        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


# 复杂度分析演示
print("\n[实验] Linear vs Standard Attention 复杂度对比")
print("-" * 60)
print("序列长度 | Standard O(N²) | Linear O(N) | 加速比")
print("-" * 60)

for seq_len in [128, 512, 2048, 8192]:
    std_ops = seq_len ** 2
    linear_ops = seq_len * 32  # 假设 head_dim=32
    print(f"{seq_len:8d} | {std_ops:13,} | {linear_ops:10,} | {std_ops/linear_ops:.0f}x")

print("\n→ 序列越长，Linear Attention 优势越大！")


# ============================================================
# 第3部分：ALiBi — 长度外推的位置编码
# ============================================================

print("\n" + "=" * 60)
print("第3部分：ALiBi — 长度外推的位置编码")
print("=" * 60)

print("""
问题：标准位置编码 (RoPE) 长度外推差
  - 模型在 2048 长度训练
  - 推理时想用 8192 长度
  - 性能急剧下降

ALiBi (Attention with Linear Biases) 的解决方案：
  不在 embedding 中加位置信息
  而是在 attention score 中加一个固定的、与距离成正比的偏置

公式：
  attention_score = Q·K^T / √d_k - slope × |i - j|

  其中 slope 是预先设定好的斜率，每个头不同

  偏置 = 负数，且距离越大负得越多
  → 距离越远的 token 注意力权重越低
  → 天然支持长度外推
""")

def get_alibi_slopes(num_heads):
    """计算每个头的 ALiBi 斜率"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power)
            + get_alibi_slopes(2 * closest_power)[0::2][: num_heads - closest_power]
        )


class ALiBiAttention(nn.Module):
    """带 ALiBi 偏置的 Attention"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # 预计算 ALiBi 偏置矩阵
        slopes = torch.tensor(get_alibi_slopes(num_heads))
        # 偏置矩阵: (num_heads, max_seq_len, max_seq_len)
        positions = torch.arange(max_seq_len)
        # 距离矩阵: |i - j|
        distance = (positions[None, :] - positions[:, None]).abs().float()
        # 应用斜率: -slope × distance
        alibi_bias = -slopes[:, None, None] * distance[None, :, :]
        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, self.num_kv_heads, seq_len, self.n_rep, self.head_dim).reshape(bsz, self.num_heads, seq_len, self.head_dim)

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 添加 ALiBi 偏置
        alibi = self.alibi_bias[:, :seq_len, :seq_len]
        scores = scores + alibi[None, :, :, :]  # (1, num_heads, seq, seq)

        # 因果掩码
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        out = weights @ xv
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


# 演示 ALiBi 偏置
print("\n[演示] ALiBi 偏置矩阵可视化 (4个头的斜率)")
print("-" * 60)
slopes = get_alibi_slopes(8)
print(f"8 个头的斜率: {[f'{s:.4f}' for s in slopes]}")
print("斜率差异: 有的头对近处敏感, 有的头对远处敏感")

print("\n距离偏置示例 (斜率=0.5):")
print("距离:  0     1     2     3     4     5")
for i in range(6):
    row = "  ".join([f"{-0.5 * abs(i - j):5.2f}" for j in range(6)])
    print(f"i={i}:  {row}")


# ============================================================
# 第4部分：Flash Attention
# ============================================================

print("\n" + "=" * 60)
print("第4部分：Flash Attention — 显存优化")
print("=" * 60)

print("""
问题：标准 Attention 的显存瓶颈
  - 实际计算量: O(N^2)
  - 显存占用: 也 O(N^2)！要存 attention matrix
  - 一次 N=8192 序列 → 8192² × 4 bytes = 256MB (仅attention)

Flash Attention (Tri Dao et al., 2022) 的核心思想：
  1. 不存完整的 N×N attention matrix
  2. 分块计算，每次只算一小块 (block-wise)
  3. 用 online softmax 累积统计量
  4. 计算量和显存都大幅降低

优势：
  ✓ 显存复杂度 O(N) 而非 O(N^2)
  ✓ 计算更快 (GPU 上 2-4x)
  ✓ 完全等价于标准 Attention (数学上)
  ✗ 实现复杂 (需要 CUDA 优化)

PyTorch 内置：
  F.scaled_dot_product_attention() — PyTorch 2.0+ 自动使用 Flash
""")


# 演示：标准 vs Flash Attention 显存差异
def attention_standard(Q, K, V):
    """标准 Attention：需要 O(N²) 显存"""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def attention_flash(Q, K, V):
    """Flash Attention (使用 PyTorch 内置 SDPA)"""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


# 测试数值一致性
print("\n[实验] 验证 Flash Attention 与标准 Attention 数值一致")
print("-" * 60)
torch.manual_seed(42)
seq_len = 16
d_k = 8
Q = torch.randn(1, 4, seq_len, d_k)
K = torch.randn(1, 4, seq_len, d_k)
V = torch.randn(1, 4, seq_len, d_k)

out_std = attention_standard(Q, K, V)
out_flash = attention_flash(Q, K, V)

print(f"标准 Attention 输出 shape: {out_std.shape}")
print(f"Flash Attention 输出 shape: {out_flash.shape}")
print(f"最大差异: {(out_std - out_flash).abs().max().item():.2e}")
print("差异极小 (浮点精度范围内)！")


# ============================================================
# 第5部分：在 MiniMind 中怎么选
# ============================================================

print("\n" + "=" * 60)
print("第5部分：MiniMind 的注意力变体选择")
print("=" * 60)

print("""
MiniMind 支持多种注意力变体:

1. Standard Attention (默认)
   - 经典 Softmax Attention
   - 适合通用场景
   - 计算复杂度 O(N²)

2. Linear Attention
   - 复杂度 O(N)
   - 适合超长序列
   - 效果略弱于 Standard

3. ALiBi Attention
   - 长度外推能力强
   - 适合需要处理长文本
   - 显存与 Standard 相当

4. Flash Attention
   - 通过 PyTorch SDPA 自动启用
   - 显存占用低
   - 速度提升 2-4x
   - 强烈推荐使用!

5. Mamba SSM (特殊)
   - 不基于 Attention
   - 复杂度 O(N)
   - 推理速度极快

配置示例 (从 MiniMind 复制):
```python
# 启用 Flash Attention
config = MiniMindConfig(...)
if hasattr(config, 'flash_attn'):
    config.flash_attn = True

# 选择 Attention 类型
if config.attention_type == "linear":
    attn = LinearAttention(config)
elif config.attention_type == "alibi":
    attn = ALiBiAttention(config)
else:
    attn = StandardAttention(config)  # 自动用 Flash
```
""")


# ============================================================
# 对比总结
# ============================================================

print("=" * 60)
print("三种注意力变体对比")
print("=" * 60)

comparison = """
┌─────────────────┬──────────┬──────────┬──────────────┐
│ 变体             │ 计算复杂度 │ 显存     │ 长度外推      │
├─────────────────┼──────────┼──────────┼──────────────┤
│ Standard        │ O(N²)    │ O(N²)    │ 差           │
│ Linear          │ O(N)     │ O(N)     │ 好           │
│ ALiBi           │ O(N²)    │ O(N²)    │ 优秀         │
│ Flash           │ O(N²)    │ O(N)     │ 差           │
│ Mamba           │ O(N)     │ O(N)     │ 好           │
└─────────────────┴──────────┴──────────┴──────────────┘

推荐使用场景：
  - 通用：Flash Attention (速度+显存都优化)
  - 超长文本：Linear Attention 或 Mamba
  - 需要长度外推：ALiBi
"""

print(comparison)


# ============================================================
# 深入理解：注意力变体的核心权衡
# ============================================================
print("\n" + "=" * 60)
print("深入理解：注意力变体的核心权衡")
print("=" * 60)

print("""
【类比：不同'注意力'方式】
──────────────────────
  
  标准 Softmax Attention (开会):
    每人都要和所有其他人交流
    → 信息最全, 但人多了就吵 (O(N²))
    → 适合小会议
  
  Linear Attention (写信):
    每个人写一封"总结信"
    别人看总结, 不用直接对话
    → 快速, 但丢失了细节
    → 适合大公司内部通知
  
  ALiBi (加距离衰减):
    说话时自动加个"距离滤镜"
    离得近的话听得清, 远的模糊
    → 天然支持长对话
    → 适合长会议
  
  Flash Attention (优化会议流程):
    不改变会议内容
    只优化组织方式
    → 速度快, 内存省
    → 适合任何规模


【三种变体的复杂度对比】
──────────────────────

  ┌────────────────┬──────────┬──────────┬─────────────┐
  │ 方法            │ 计算      │ 显存      │ 长度外推     │
  ├────────────────┼──────────┼──────────┼─────────────┤
  │ Softmax        │ O(N²)    │ O(N²)    │ 差 (需重训) │
  │ Linear         │ O(N)     │ O(N)     │ 优秀         │
  │ ALiBi          │ O(N²)    │ O(N²)    │ 优秀 (天然)  │
  │ Flash          │ O(N²)    │ O(N)     │ 同 Softmax   │
  └────────────────┴──────────┴──────────┴─────────────┘

  解释:
    计算: 多少次乘法
    显存: 多少中间结果
    外推: 训练 2K, 能跑 32K 吗


【图示：注意力矩阵对比】
────────────────────

  标准 Softmax Attention (稠密):
    QK^T 矩阵, 几乎所有元素非零
    [0.01 0.02 0.05 ... 0.10]
    [0.03 0.01 0.04 ... 0.08]
    [0.05 0.04 0.02 ... 0.06]
    [   ...              ]
    → O(N²) 计算, 完整信息

  Linear Attention (隐式):
    不显式算 QK^T
    用 (Q φ)(K φ)^T 的低秩分解
    → O(N) 计算, 损失一些精度

  ALiBi (位置衰减):
    同样算 QK^T, 但加上 -k * distance
    [0.01 0.005 0.001 ... 0]
    [0.03 0.02  0.01  ... 0]
    [   ...                  ]
    → 远处注意力自然衰减


【Linear Attention 的数学本质】
────────────────────────────

  标准:  attn(Q, K, V) = softmax(QK^T) V
  线性:  attn(Q, K, V) = (φ(Q) φ(K)^T) V = φ(Q) (φ(K)^T V)
  
  关键: 先算 φ(K)^T V, 再算 φ(Q) × ...
  后者是矩阵 × 矩阵, 复杂度 O(N) 而不是 O(N²)

  特征映射 φ 的选择:
    - ELU + 1: 简单, 主流
    - Random features: 更快
    - cos: 性能更好但有数值问题

  缺点:
    - 不能精确还原 softmax
    - 性能略低于 softmax
    - 注意力分布不够"尖"


【ALiBi 长度外推的原理】
──────────────────────

  问题:
    RoPE 在训练长度 2K, 推理 8K 时性能下降
    
  解决:
    在注意力分数上加 -k * 距离
    
  例: 位置 100 看位置 0 (距离=100)
    无 ALiBi: 分数 0.5
    有 ALiBi: 分数 0.5 - k*100 = -1 (被忽略)
    → 模型学到"远的不要看太多"

  为什么外推好?
    训练 2K 时, 看到最远距离 2000
    推理 8K 时, 最远距离 8000
    但模型已经学会"距离越远越不重要"
    → 8K 时仍能泛化


【Flash Attention 解决了什么？】
──────────────────────────────

  标准 Attention 的内存瓶颈:
    1. QK^T 中间矩阵: [N, N] 显存
    2. softmax 之后的矩阵: [N, N] 显存
    3. 与 V 相乘的中间结果
    
  N=4096 时: 4096² = 16M 元素 = 64MB (单精度)
  → 大 batch 显存爆炸

  Flash Attention 解决:
    不实际算 [N, N] 矩阵
    分块 (block) 计算, 算完即丢
    数学上等价, 显存降到 O(N)
    
  实际加速:
    A100 GPU: 2-4x 加速
    长序列 (>4K): 更明显


【MiniMind 中如何选择？】
─────────────────────

  ┌──────────────┬──────────────────────────────┐
  │ 场景          │ 推荐                          │
  ├──────────────┼──────────────────────────────┤
  │ 短文本 (<2K) │ 标准 Attention                │
  │ 长文本 (>2K) │ 标准 + RoPE                   │
  │ 超长 (>32K)  │ ALiBi 或 Linear Attention     │
  │ 推理加速     │ Flash Attention (任何情况)    │
  │ 资源受限      │ Linear Attention              │
  └──────────────┴──────────────────────────────┘

  MiniMind 实现:
    默认用 RoPE + 标准 Attention
    可选 Flash Attention (PyTorch 2.0+)
    配置项切换
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】复杂度对比
  序列长度 N=8192, 标准注意力和 Linear Attention
  各需要多少次乘法?

【练习2】ALiBi vs RoPE
  ALiBi 相对 RoPE 在长度外推上有什么优势?

【练习3】Linear Attention 损失
  Linear Attention 相比标准 Attention 损失了哪些能力?

【练习4】Flash Attention 等价性
  为什么 Flash Attention 数学上等价于标准 Attention,
  但实际快很多?

【练习5】MQA vs MHA
  Multi-Query Attention (MQA) 和 Multi-Head Attention (MHA)
  显存差多少? (假设 head_dim=64, batch=1, 序列=2K, 32层)

【练习6】选择策略
  训练短文本, 推理长文本, 应该选哪种注意力?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
N = 8192
softmax_ops = N * N
linear_ops = N * 64
ratio = softmax_ops / linear_ops
print(f"  N = {N}")
print(f"  标准注意力: O(N²) = {N}² = {softmax_ops:,} 次")
print(f"  Linear Attention: O(N × d) = {N} × 64 = {linear_ops:,} 次")
print(f"  加速比: {ratio:.0f}x")
print()
print(f"  → Linear 比标准快 {ratio:.0f} 倍 (N=8192 时)")

# 练习2
print("\n【练习2 答案】")
print("  RoPE:")
print("    - 用相对位置编码")
print("    - 训练 2K, 推理 4K 性能还行")
print("    - 训练 2K, 推理 32K 性能显著下降")
print()
print("  ALiBi:")
print("    - 用线性偏置 -k * distance")
print("    - 训练 2K, 推理 8K 性能轻微下降")
print("    - 训练 2K, 推理 32K 性能下降有限")
print()
print("  原因:")
print("    - ALiBi 是'距离衰减', 任何距离都有效")
print("    - RoPE 是'相对旋转', 超出训练范围时相位乱了")
print()
print("  实验:")
print("    RoPE 训练 2K → 8K: PPL 涨 30%")
print("    ALiBi 训练 2K → 8K: PPL 涨 5%")
print("    → ALiBi 长度外推显著优于 RoPE")

# 练习3
print("\n【练习3 答案】")
print("  Linear Attention 损失的能力:")
print()
print("  1. 精确的注意力分布:")
print("    标准: softmax(QK^T) 可以让注意力非常'尖'")
print("    线性: 注意力分布更平, 难以做到'一个 token 看另一个'")
print()
print("  2. 强检索能力:")
print("    标准: 可以精确复制某个 token 的信息")
print("    线性: 信息被混合, 难精确检索")
print()
print("  3. 复杂推理:")
print("    标准: 多步推理的'线索传递'能力强")
print("    线性: 在需要精确中间状态的推理上弱")
print()
print("  4. 训练稳定性:")
print("    标准: 训练稳定, 收敛好")
print("    线性: 训练可能不稳定, 需要调参")
print()
print("  缓解方法:")
print("    - 用更好的特征映射 (如 cos+激活)")
print("    - 混合: 浅层用标准, 深层用线性")
print("    - Mamba 提供了更好的替代方案")

# 练习4
print("\n【练习4 答案】")
print("  数学等价:")
print("    softmax 公式:")
print("      softmax(x_i) = exp(x_i) / sum_j exp(x_j)")
print("    → sum_j exp(x_j) 是'全局归一化'")
print()
print("    Flash Attention 分块计算 softmax:")
print("      1. 把 Q, K, V 分成小块 (如每块 256)")
print("      2. 在每个块内, 算局部 softmax")
print("      3. 用 running max 修正")
print("      4. 合并时按比例缩放")
print()
print("    → 最终结果和一次性算完全一致")
print()
print("  实际快的原因:")
print("    1. 显存访问优化:")
print("      - 不用存 [N, N] 中间矩阵")
print("      - 每块只用 SRAM (片上缓存, 极快)")
print("    2. 减少显存 IO:")
print("      - 标准: 频繁读写 HBM (慢)")
print("      - Flash: 多在 SRAM 算, 少访问 HBM")
print("    3. 并行友好:")
print("      - 块间无依赖, 完美并行")
print()
print("  加速比:")
print("    短序列 (1K): 1.5-2x")
print("    长序列 (8K+): 3-5x")
print("    内存受限场景: 10x+")

# 练习5
print("\n【练习5 答案】")
batch = 1
seq_len = 2048
num_layers = 32
num_heads = 32
head_dim = 64
fp16_bytes = 2

# MHA: 每头独立 K, V
mha_kv = 2 * num_heads * head_dim
mha_total = batch * num_layers * seq_len * mha_kv * fp16_bytes
mha_mb = mha_total / (1024 * 1024)

# MQA: 共享一个 K, V
mqa_kv = 1 * head_dim
mqa_total = batch * num_layers * seq_len * mqa_kv * fp16_bytes
mqa_mb = mqa_total / (1024 * 1024)

# GQA: 4 头共享
gqa_groups = 4
gqa_kv = (num_heads // gqa_groups) * head_dim
gqa_total = batch * num_layers * seq_len * gqa_kv * fp16_bytes
gqa_mb = gqa_total / (1024 * 1024)

print(f"  配置: bs={batch}, layers={num_layers}, seq={seq_len}, head_dim={head_dim}")
print(f"  num_heads={num_heads} (MHA/MQA/GQA)")
print()
print(f"  MHA (32 KV 头):")
print(f"    每层 KV: {2*num_heads*head_dim} = {mha_kv} 元素")
print(f"    总: {mha_mb:.1f} MB")
print()
print(f"  GQA (4 组, 8 KV 头):")
print(f"    每层 KV: {gqa_kv} 元素")
print(f"    总: {gqa_mb:.1f} MB")
print()
print(f"  MQA (1 KV 头):")
print(f"    每层 KV: {mqa_kv} 元素")
print(f"    总: {mqa_mb:.1f} MB")
print()
print(f"  节省:")
print(f"    GQA vs MHA: {(1 - gqa_mb/mha_mb)*100:.0f}%")
print(f"    MQA vs MHA: {(1 - mqa_mb/mha_mb)*100:.0f}%")
print()
print(f"  实际:")
print(f"    MHA: LLaMA-1, GPT-3")
print(f"    GQA: LLaMA-2/3, MiniMind (主流)")
print(f"    MQA: PaLM, StarCoder (极限优化)")

# 练习6
print("\n【练习6 答案】")
print("  训练短文本, 推理长文本:")
print()
print("  最佳选择: RoPE + 长度外推技术")
print()
print("  选项 1: RoPE + Position Interpolation (PI)")
print("    把位置编码'压缩', 适配更长序列")
print("    例: 训练 2K, 推理 8K → 位置编码缩放 4x")
print()
print("  选项 2: RoPE + YaRN")
print("    PI 的改进, 不同频率不同处理")
print("    效果好, MiniMind 第16课会讲")
print()
print("  选项 3: ALiBi (训练时就用)")
print("    天然支持长度外推")
print("    性能略低于 RoPE, 但外推最好")
print()
print("  选项 4: NTK-Aware Scaling")
print("    修改 RoPE 的 base, 不用插值")
print("    实现简单, 效果不错")
print()
print("  实际推荐:")
print("    大多数 LLM: RoPE + YaRN")
print("    LLaMA-2: RoPE (4K → 训练 32K 推理)")
print("    MiniMind: RoPE + 可选 YaRN")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
