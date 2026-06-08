"""
第5课：Attention — 核心机制：词与词如何相互关注？
=====================================================

Attention 是 Transformer 的灵魂。
一句话概括：每个词向其他所有词"提问"，根据相关性收集信息。

核心公式：
  Attention(Q, K, V) = softmax(Q·K^T / √d) · V

拆解：
  Q (Query):  "我在找什么？" — 每个词的查询向量
  K (Key):    "我有什么？"   — 每个词的键向量
  V (Value):  "我的内容是"   — 每个词的值向量

  1. Q·K^T:    计算每对词的相关性分数
  2. /√d:      缩放，防止分数过大
  3. softmax:  归一化为概率分布（注意力权重）
  4. ·V:       用权重加权求和，得到每个词的新表示

MiniMind 使用 GQA (Grouped Query Attention)：
  多个 Q 头共享一组 K/V 头，减少 KV 缓存的内存开销。

运行: python lessons/lesson05_attention.py
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 第一步：Attention 的直觉理解
# ============================================================

print("=" * 60)
print("实验1：Attention 的直觉 — 信息检索")
print("=" * 60)

# 想象你在图书馆找书：
# Q = 你的需求："我想找关于猫的书"
# K = 每本书的标签："猫"、"狗"、"烹饪"
# V = 每本书的内容

# 用向量模拟
query = torch.tensor([1.0, 0.0])      # 需求：关注"猫"维度
key_cat = torch.tensor([0.9, 0.1])     # 标签：主要是猫
key_dog = torch.tensor([0.1, 0.9])     # 标签：主要是狗
key_cook = torch.tensor([0.0, 0.0])    # 标签：都不相关

# 计算相关性分数（点积）
score_cat = torch.dot(query, key_cat)
score_dog = torch.dot(query, key_dog)
score_cook = torch.dot(query, key_cook)

print(f"查询: {query.tolist()}")
print(f"与'猫'的相关性: {score_cat:.2f}")
print(f"与'狗'的相关性: {score_dog:.2f}")
print(f"与'烹饪'的相关性: {score_cook:.2f}")
print("→ 相关性越高，获得的注意力权重越大")


# ============================================================
# 第二步：手动实现单头 Attention
# ============================================================

def single_head_attention(Q, K, V, mask=None):
    """手动实现单头注意力

    Args:
        Q: [seq_len, d_k] 查询矩阵
        K: [seq_len, d_k] 键矩阵
        V: [seq_len, d_v] 值矩阵
        mask: 可选的掩码

    Returns:
        output: [seq_len, d_v] 注意力输出
        weights: [seq_len, seq_len] 注意力权重
    """
    d_k = Q.shape[-1]

    # Step 1: 计算注意力分数
    scores = Q @ K.T / math.sqrt(d_k)

    # Step 2: 应用掩码（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: softmax 归一化
    weights = torch.softmax(scores, dim=-1)

    # Step 4: 加权求和
    output = weights @ V

    return output, weights


# 实验2：单头注意力
print("\n" + "=" * 60)
print("实验2：单头注意力计算")
print("=" * 60)

seq_len = 4
d_k = 8

torch.manual_seed(42)
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

output, weights = single_head_attention(Q, K, V)

print(f"Q 形状: {Q.shape} (4个词，每个8维查询)")
print(f"K 形状: {K.shape} (4个词，每个8维键)")
print(f"V 形状: {V.shape} (4个词，每个8维值)")
print(f"\n注意力权重形状: {weights.shape} (4×4，每行是一个词对其他词的关注度)")
print(f"注意力权重:\n{weights}")
print(f"\n每行之和: {weights.sum(dim=-1).tolist()} (softmax后每行=1)")
print(f"输出形状: {output.shape} (4个词的新表示)")


# ============================================================
# 第三步：因果掩码 — 语言模型只能看过去
# ============================================================

print("\n" + "=" * 60)
print("实验3：因果掩码 (Causal Mask)")
print("=" * 60)

# 语言模型是自回归的：预测第 t 个词时，只能看前 t-1 个词
# 用下三角掩码实现：
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(f"因果掩码 (1=可见, 0=不可见):\n{causal_mask.int()}")

print("\n解释:")
print("  位置0只能看位置0")
print("  位置1能看位置0,1")
print("  位置2能看位置0,1,2")
print("  位置3能看位置0,1,2,3")

# 带因果掩码的注意力
output_causal, weights_causal = single_head_attention(Q, K, V, mask=causal_mask)
print(f"\n带掩码的注意力权重:\n{weights_causal}")
print("→ 上三角全为0，每个词只关注自己和之前的词")


# ============================================================
# 第四步：多头注意力 (Multi-Head Attention)
# ============================================================

print("\n" + "=" * 60)
print("实验4：多头注意力 — 让模型从不同角度关注")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    """手动实现多头注意力"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        # 每个头有自己的 Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape

        # 线性投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 拆分成多个头: [bsz, seq_len, hidden_size] → [bsz, seq_len, num_heads, head_dim]
        Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim)
        K = K.view(bsz, seq_len, self.num_heads, self.head_dim)
        V = V.view(bsz, seq_len, self.num_heads, self.head_dim)

        # 转置: [bsz, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 注意力计算
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores.float(), dim=-1).type_as(Q)
        output = weights @ V

        # 合并头: [bsz, num_heads, seq_len, head_dim] → [bsz, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        output = self.o_proj(output)

        return output, weights


# 实验
hidden_size = 64
num_heads = 4
seq_len = 8
batch_size = 2

mha = MultiHeadAttention(hidden_size, num_heads)
x = torch.randn(batch_size, seq_len, hidden_size)
causal = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

output, weights = mha(x, mask=causal)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
print(f"  num_heads={num_heads}, head_dim={hidden_size//num_heads}")
print(f"\n每个头关注不同的模式:")
for h in range(num_heads):
    print(f"  头{h}: 位置0对位置0-3的权重 = {weights[0, h, 0, :4].detach().tolist()}")


# ============================================================
# 第五步：GQA — Grouped Query Attention
# ============================================================

print("\n" + "=" * 60)
print("实验5：GQA — 分组查询注意力")
print("=" * 60)

# 标准 MHA: 每个Q头有独立的K,V头
#   8个Q头 → 8个K头 + 8个V头
# GQA: 多个Q头共享一组K,V头
#   8个Q头 → 2个K头 + 2个V头 (每4个Q头共享1组KV)

class GroupedQueryAttention(nn.Module):
    """GQA: 多个 Q 头共享一组 KV 头"""

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads  # 每组KV头对应的Q头数
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

        # 扩展 KV 头以匹配 Q 头数量
        # [bsz, seq_len, num_kv_heads, head_dim] → [bsz, seq_len, num_heads, head_dim]
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


# 对比 MHA vs GQA
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

print(f"MHA: {num_heads}个Q头, {num_kv_heads_mha}个KV头, 参数量={mha_params:,}")
print(f"GQA: {num_heads}个Q头, {num_kv_heads_gqa}个KV头, 参数量={gqa_params:,}")
print(f"GQA 参数量减少: {(1 - gqa_params/mha_params)*100:.1f}%")
print(f"\nGQA 的优势:")
print(f"  1. KV 缓存更小（只需存 {num_kv_heads_gqa} 组而非 {num_kv_heads_mha} 组）")

# 【新增】GQA 的具体数字对比：KV Cache 节省了多少？
print("\n" + "-" * 50)
print("【GQA 的具体数字对比：KV Cache 节省了多少？】")
print("-" * 50)
print("""
以 LLaMA-2-70B 为例（类似 MiniMind 的架构）：
  hidden_size = 8192
  num_heads (Q) = 64
  head_dim = 128

  MHA（所有头独立）：
    KV 头数 = 64
    每个token的KV Cache = 2 × 64 × 128 = 16,384 个float16
    = 32 KB / token

  GQA（8组KV头）：
    KV 头数 = 8
    每个token的KV Cache = 2 × 8 × 128 = 2,048 个float16
    = 4 KB / token
    节省 = (64-8)/64 = 87.5%！

  实际影响：
    生成4096个token时：
    MHA: 4096 × 32 KB = 128 MB（仅KV Cache）
    GQA: 4096 × 4 KB  =  16 MB（仅KV Cache）
    → 节省 112 MB，可以多生成7倍的token！

  MiniMind 的情况：
    num_heads = 12, num_kv_heads = 4
    KV Cache 节省 = (12-4)/12 = 66.7%
""")
print(f"  2. 推理更快（KV 缓存读取量减少 {(1 - num_kv_heads_gqa/num_kv_heads_mha)*100:.0f}%）")
print(f"  3. 效果接近 MHA（共享的KV头仍能捕捉关键信息）")


# ============================================================
# 第5.5步：QK-Norm — 稳定注意力训练的关键技巧
# ============================================================

print("\n" + "=" * 60)
print("实验5.5：QK-Norm — 稳定注意力训练的关键技巧")
print("=" * 60)

print("""
问题：当模型变大、训练变长时，Q·K^T 的数值可能爆炸
  → softmax 输入过大 → 输出趋向 one-hot
  → 梯度消失 → 训练不稳定甚至崩溃

解决方案：对 Q 和 K 做归一化 (QK-Norm)
  在计算注意力分数之前，对每个头的 Q 和 K 单独做 RMSNorm

  标准 Attention:  scores = Q @ K^T / sqrt(d_k)
  QK-Norm:        scores = RMSNorm(Q) @ RMSNorm(K)^T / sqrt(d_k)

MiniMind 原始项目使用了 QK-Norm，这是现代 LLM 的标配技巧。
""")

class RMSNormForQK(nn.Module):
    """用于 QK-Norm 的 RMSNorm（与第3课相同）"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class GroupedQueryAttentionWithQKNorm(nn.Module):
    """带 QK-Norm 的 GQA — 与 MiniMind 原始实现一致"""

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

print(f"无 QK-Norm 输出范围: [{out_without.min().item():.4f}, {out_without.max().item():.4f}]")
print(f"有 QK-Norm 输出范围: [{out_with.min().item():.4f}, {out_with.max().item():.4f}]")

qknorm_extra = sum(p.numel() for p in gqa_with_qknorm.q_norm.parameters()) + sum(p.numel() for p in gqa_with_qknorm.k_norm.parameters())
print(f"\nQK-Norm 额外参数: {qknorm_extra} (head_dim × 2 = {hidden_size // num_heads} × 2)")
print("→ 参数量增加极少，但训练稳定性显著提升")


# ============================================================
# 第六步：MiniMind 的 Attention 配置
# ============================================================

print("\n" + "=" * 60)
print("实验6：MiniMind 的 Attention 配置")
print("=" * 60)

configs = {
    "MiniMind (小)": {"num_heads": 12, "num_kv_heads": 4, "head_dim": 64, "hidden_size": 768},
    "MiniMind (中)": {"num_heads": 16, "num_kv_heads": 4, "head_dim": 64, "hidden_size": 1024},
}

for name, cfg in configs.items():
    n_rep = cfg["num_heads"] // cfg["num_kv_heads"]
    kv_cache_per_token = cfg["num_kv_heads"] * cfg["head_dim"] * 2  # K+V
    print(f"{name}:")
    print(f"  Q头数: {cfg['num_heads']}, KV头数: {cfg['num_kv_heads']}, 每组{n_rep}个Q头共享1组KV")
    print(f"  每个token的KV缓存: {kv_cache_per_token} 个float = {kv_cache_per_token*4/1024:.1f} KB")
    print(f"  2048个token的KV缓存: {kv_cache_per_token*2048*4/1024/1024:.2f} MB")
    print()


# ============================================================
# 第七步：Attention 的完整计算流程
# ============================================================

print("=" * 60)
print("实验7：从输入到输出的完整流程")
print("=" * 60)

# 用 MiniMind 小模型配置
hidden_size = 768
num_heads = 12
num_kv_heads = 4
head_dim = 64
seq_len = 16
batch_size = 1

# 创建各组件
q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

# 模拟输入（经过 Embedding + RMSNorm 后的向量）
x = torch.randn(batch_size, seq_len, hidden_size)

# Step 1: 线性投影
xq = q_proj(x)
xk = k_proj(x)
xv = v_proj(x)
print(f"1. 线性投影: x[{x.shape}] → Q[{xq.shape}], K[{xk.shape}], V[{xv.shape}]")

# Step 2: reshape 为多头格式
xq = xq.view(batch_size, seq_len, num_heads, head_dim)
xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim)
xv = xv.view(batch_size, seq_len, num_kv_heads, head_dim)
print(f"2. Reshape: Q[{xq.shape}], K[{xk.shape}], V[{xv.shape}]")

# Step 3: 扩展 KV 头 (GQA)
n_rep = num_heads // num_kv_heads
xk_exp = xk[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, num_heads, head_dim)
xv_exp = xv[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, num_heads, head_dim)
print(f"3. GQA扩展: K[{xk_exp.shape}], V[{xv_exp.shape}]")

# Step 4: 转置为 [batch, heads, seq, dim]
xq = xq.transpose(1, 2)
xk_exp = xk_exp.transpose(1, 2)
xv_exp = xv_exp.transpose(1, 2)
print(f"4. 转置: Q[{xq.shape}], K[{xk_exp.shape}], V[{xv_exp.shape}]")

# Step 5: 注意力分数
scores = xq @ xk_exp.transpose(-2, -1) / math.sqrt(head_dim)
print(f"5. 注意力分数: [{scores.shape}]")

# Step 6: 因果掩码
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
scores_masked = scores + causal_mask
print(f"6. 应用因果掩码: 上三角→-inf")

# Step 7: softmax
attn_weights = torch.softmax(scores_masked.float(), dim=-1)
print(f"7. Softmax: 注意力权重 [{attn_weights.shape}]")

# Step 8: 加权求和
attn_output = attn_weights @ xv_exp
print(f"8. 加权求和: [{attn_output.shape}]")

# Step 9: 合并头 + 输出投影
attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
output = o_proj(attn_output)
print(f"9. 合并头+投影: [{output.shape}]")

print(f"\n输入: [{batch_size}, {seq_len}, {hidden_size}]")
print(f"输出: [{output.shape}]")
print("Attention 不改变形状，只改变内容！")


# ============================================================
# 第八步：注意力权重可视化
# ============================================================

print("\n" + "=" * 60)
print("实验8：注意力权重模式")
print("=" * 60)

# 用一个简单例子观察注意力模式
seq_len = 6
d_k = 4

# 模拟一个句子的注意力："猫 坐 在 垫子 上 睡觉"
# 期望："猫"关注"睡觉"，"在"关注"垫子"
words = ["猫", "坐", "在", "垫子", "上", "睡觉"]

torch.manual_seed(7)
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

causal = torch.tril(torch.ones(seq_len, seq_len))
_, weights = single_head_attention(Q, K, V, mask=causal)

print(f"句子: {' '.join(words)}")
print(f"\n注意力权重矩阵 (每行=一个词对其他词的关注度):")
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

print("\n*** = 高关注, ** = 中关注, * = 低关注, . = 几乎不关注")
print("因果掩码确保每个词只看自己和之前的词（下三角）")


# ============================================================
# 深入理解：Attention 的直觉与类比
# ============================================================
print("\n" + "=" * 60)
print("深入理解：Attention 的直觉与类比")
print("=" * 60)

print("""
【类比1：图书馆找书】
─────────────────────
想象你在图书馆找书:
  - Q (Query)   = 你的"问题"或"兴趣" (比如"我想学 Python")
  - K (Key)     = 每本书的"标签"或"目录" (书名、关键词)
  - V (Value)   = 每本书的"实际内容"

  流程:
    1. 用你的 Q 去和所有书的 K 比较 (内积)
    2. 相似度高的书得到高权重 (softmax)
    3. 按权重把所有书的内容 V 混合起来
    4. 混合结果就是"你想要的知识"

  → Attention 就是"按相关性加权求和"


【类比2：会议讨论】
─────────────────
开会时, 每个人发言, 但你只会仔细听相关的人:
  - Q = 你关心的话题
  - K = 每个人发言的主题
  - V = 每个人发言的详细内容
  - 内积 = "他的话跟我的话题相关度"
  - softmax = 决定我"分多少注意力给他"

  在 Transformer 中:
    每个词都是一个"发言人"
    同时是听众 (用 Q 提问)
    也是发言者 (用 V 贡献内容)


【图示：Attention 计算流程】
──────────────────────────

   输入: Q, K, V
       │
       ▼
   ┌────────────────┐
   │ Q @ K^T        │  ← 计算相似度
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ ÷ sqrt(d_k)    │  ← 缩放, 防止 softmax 饱和
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ + Mask (opt)   │  ← 因果掩码
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ Softmax (行)   │  ← 归一化为概率
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ @ V            │  ← 加权求和
   └───────┬────────┘
           │
           ▼
       输出

每一行 (每个 query) 的权重和 = 1
  → 每个位置"分配 1 个单位的注意力"给所有位置


【为什么除以 sqrt(d_k)?】
─────────────────────────
  问题: 当 d_k 大时, Q·K^T 的方差也会变大
  方差大 → softmax 趋向 one-hot (只关注一个位置)
  → 梯度小, 训练困难

  数学上:
    Q, K 元素 ~ N(0, 1) (假设)
    Q·K^T = sum_i Q_i * K_i
    Var(Q·K^T) = d_k
    标准差 = sqrt(d_k)
    → 除以 sqrt(d_k) 后, 方差归一化到 1

  实验: d_k=64, 不缩放时, softmax 输出接近 one-hot
        d_k=64, 缩放后,   softmax 输出比较平滑

【多头注意力的几何意义】
────────────────────────
每个头学习不同的"关注模式":
  - 头 1: 关注"主谓关系" (动词看主语)
  - 头 2: 关注"指代关系" (代词看名词)
  - 头 3: 关注"长距离依赖" (首尾呼应)
  - 头 4: 关注"局部语法" (相邻词)
  - ...

  类比: 一个团队合作, 每人专攻一个方面
  → 多人合作比单干更全面!
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】Q, K, V 的来源
  Q, K, V 都来自输入 x, 为什么要做 3 次不同的线性投影?

【练习2】缩放因子
  Attention 中除以 sqrt(d_k), 如果不除会怎样?

【练习3】因果掩码
  生成第 t 个 token 时, 能看到位置 t 之前的还是之后的内容?

【练习4】多头 vs 单头
  同样是 d 维向量, 多头注意力有什么优势?

【练习5】注意力权重和
  softmax(QK^T) 每一行的和是多少? 物理含义是什么?

【练习6】自注意力 vs 交叉注意力
  自注意力的 Q, K, V 都来自同一个输入
  交叉注意力的 Q 来自一个输入, K 和 V 来自另一个
  各自用在什么场景?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  Q, K, V 是同一个向量的 3 个不同'视角'")
print("  - W_Q 投影: 学到'这个 token 在关注什么' (查询)")
print("  - W_K 投影: 学到'这个 token 能被什么关注' (被查询)")
print("  - W_V 投影: 学到'这个 token 想传递什么信息' (内容)")
print()
print("  如果用同一个投影 (Q=K=V=x), 注意力会退化为:")
print("  softmax(x·x^T)·x, 缺少'问-答'的解耦")
print("  → 模型无法区分'我在找什么'和'我能提供什么'")
print("  → 多头也是同理, 每个头学不同的 W_Q, W_K, W_V")

# 练习2
print("\n【练习2 答案】")
print("  如果不除以 sqrt(d_k):")
print("  - 大 d_k 时, QK^T 数值过大")
print("  - softmax 趋向 one-hot (只关注一个位置)")
print("  - 梯度消失 (其他位置梯度接近 0)")
print("  - 训练不稳定, 模型难以学到多样化的关注")
print()
print("  演示:")
import torch
import torch.nn.functional as F
for d_k in [4, 16, 64, 256]:
    q = torch.randn(1, 1, 1, d_k)
    k = torch.randn(1, 1, 5, d_k)
    scores = (q @ k.transpose(-2, -1)).squeeze()
    print(f"    d_k={d_k}: 不缩放 scores={scores.tolist()}")
    weights = F.softmax(scores, dim=-1)
    print(f"            softmax={weights.tolist()}")
    print(f"            最大值={weights.max():.4f} (越大越 one-hot)")

# 练习3
print("\n【练习3 答案】")
print("  只能看到位置 t 之前的内容 (包括自己)")
print("  因果掩码 (causal mask) 把上三角设为 -inf")
print("  softmax(-inf) = 0, 未来位置不参与注意力")
print()
print("  为什么? 语言模型是自回归的")
print("  训练时已知完整句子, 但推理时是逐个生成")
print("  训练时如果让模型看到未来, 推理时就不知道未来")
print("  → 训练-推理不一致, 训练失败")

# 练习4
print("\n【练习4 答案】")
print("  多头注意力的优势:")
print("  1. 关注不同方面: 一个头关注语法, 另一个关注语义")
print("  2. 关注不同位置: 一个头看近距离, 另一个看远距离")
print("  3. 关注不同关系: 主谓、动宾、指代等不同模式")
print()
print("  数学上: 多头相当于把 d 维向量拆成 h 份, 每份 d/h 维")
print("  每个头独立做 attention, 最后拼接")
print("  → 等价于在低秩子空间中做 attention, 表达力更强")

# 练习5
print("\n【练习5 答案】")
print("  每一行的和 = 1")
print("  softmax 把每一行归一化为概率分布")
print()
print("  物理含义: '我把 1 个单位的注意力分配给所有位置'")
print("  - 自己 = 0.5 (主要关注自己)")
print("  - 关键词 = 0.3 (重要内容)")
print("  - 无关词 = 0.1 + 0.05 + 0.05 = 0.2 (背景)")
print("  → 总和 = 1.0")
print()
print("  这种'概率'解释让 attention 非常优雅")
print("  还可以可视化 attention map 理解模型在关注什么")

# 练习6
print("\n【练习6 答案】")
print("  自注意力 (Self-Attention):")
print("    Q, K, V 都来自同一个序列")
print("    用途: 让序列内部的信息相互流动")
print("    例: GPT (文本生成), BERT (文本理解)")
print()
print("  交叉注意力 (Cross-Attention):")
print("    Q 来自一个序列, K, V 来自另一个序列")
print("    用途: 让一个序列去'查询'另一个序列")
print("    例: 翻译 (中文 Q 查询英文 K, V)")
print("    例: 文生图 (文本 Q 查询图像 K, V)")
print("    例: RAG (问题 Q 查询文档 K, V)")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. Attention = softmax(Q·K^T/√d)·V，让词与词相互交换信息
2. Q=查询，K=键，V=值，类似信息检索
3. 因果掩码确保语言模型只能看过去，不能看未来
4. 多头注意力让模型从不同角度关注
5. GQA 让多个Q头共享KV头，减少KV缓存开销
6. MiniMind 使用 GQA（如12个Q头共享4个KV头）
7. QK-Norm 对 Q 和 K 做归一化，防止注意力分数爆炸，稳定训练

数据流至此：
  Token IDs → Embedding → RMSNorm → RoPE → Attention → ...
  [42,108]   [0.1,-0.3]  [0.8,0.2]  旋转Q/K  上下文感知的向量

下一步 → lesson06_ffn.py：前馈网络——注意力之后的记忆提炼
""")
