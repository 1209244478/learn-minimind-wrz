"""
第4课：RoPE — 模型如何知道词的位置？
=====================================

"猫吃鱼" 和 "鱼吃猫" 包含相同的词，但意思完全不同。
模型必须知道每个词的位置，才能理解语序。

最朴素的想法：给每个位置一个编号（0, 1, 2, ...），加到向量上。
问题：编号是绝对值，模型难以理解"位置1和位置3的距离=2"这种相对关系。

RoPE（Rotary Position Embedding，旋转位置编码）的思路：
  不加位置信息，而是用"旋转"来编码位置！
  把向量看作二维平面上的点，位置 m 的向量旋转 mθ 角度。

关键性质：
  1. 内积只依赖相对位置差（m-n），天然编码相对位置
  2. 远距离衰减：位置差越大，内积越小
  3. 无需额外参数，纯数学运算

运行: python lessons/lesson04_rope.py
"""

import torch
import math


# ============================================================
# 第一步：为什么需要位置编码？
# ============================================================

print("=" * 60)
print("实验1：没有位置信息，模型分不清语序")
print("=" * 60)

# 假设3个词的嵌入向量
word_cat = torch.tensor([1.0, 0.5])
word_eat = torch.tensor([0.3, 0.8])
word_fish = torch.tensor([0.7, 0.2])

# "猫吃鱼" 和 "鱼吃猫" 的词相同，但顺序不同
# 如果没有位置信息，模型看到的向量集合完全一样！
sentence1 = [word_cat, word_eat, word_fish]   # 猫吃鱼
sentence2 = [word_fish, word_eat, word_cat]   # 鱼吃猫

print("猫吃鱼:", [w.tolist() for w in sentence1])
print("鱼吃猫:", [w.tolist() for w in sentence2])
print("词的集合相同，但意思完全不同！")
print("→ 模型需要知道每个词在哪个位置")


# ============================================================
# 第二步：2D 旋转 — RoPE 的核心直觉
# ============================================================
# 在二维平面上，旋转一个向量 θ 角度的矩阵：
#   R(θ) = [cos θ, -sin θ]
#          [sin θ,  cos θ]
#
# 位置 m 的向量旋转 mθ：
#   x_m' = R(mθ) · x_m
#
# 两个位置 m 和 n 的向量内积：
#   <R(mθ)·q, R(nθ)·k> = <q, k> · cos((m-n)θ)
#
# 只依赖 (m-n)！这就是相对位置编码！

print("\n" + "=" * 60)
print("实验2：2D 旋转编码位置")
print("=" * 60)

def rotate_2d(vec, angle):
    """2D 旋转：把向量旋转 angle 弧度"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y = vec[0], vec[1]
    return torch.tensor([x * cos_a - y * sin_a, x * sin_a + y * cos_a])

theta = 0.5  # 基础角度

# 位置0的向量不旋转
q_pos0 = rotate_2d(word_cat, 0 * theta)
# 位置1的向量旋转 θ
q_pos1 = rotate_2d(word_cat, 1 * theta)
# 位置2的向量旋转 2θ
q_pos2 = rotate_2d(word_cat, 2 * theta)

print(f"原始向量:     {word_cat.tolist()}")
print(f"位置0 (0θ):   {q_pos0.tolist()}")
print(f"位置1 (1θ):   {q_pos1.tolist()}")
print(f"位置2 (2θ):   {q_pos2.tolist()}")

# 验证：内积只依赖相对位置差
dot_01 = torch.dot(q_pos0, q_pos1)
dot_12 = torch.dot(q_pos1, q_pos2)
print(f"\n位置0和1的内积: {dot_01:.4f}")
print(f"位置1和2的内积: {dot_12:.4f}")
print(f"两者接近（因为相对距离都是1）: {abs(dot_01 - dot_12) < 0.01}")

dot_02 = torch.dot(q_pos0, q_pos2)
print(f"位置0和2的内积: {dot_02:.4f} (距离2，内积更小)")
print("→ 距离越远，内积越小（远距离衰减）")


# ============================================================
# 第三步：推广到高维 — 分组旋转
# ============================================================
# 一个 d 维向量，可以分成 d/2 组二维向量：
#   [x0, x1, x2, x3, ...] → [(x0,x1), (x2,x3), ...]
# 每组用不同的频率 θ_i 旋转：
#   θ_i = 1 / (base^(2i/d))
#
# 低维组用大角度（高频），捕捉局部位置差异
# 高维组用小角度（低频），捕捉全局位置信息

print("\n" + "=" * 60)
print("实验3：不同维度用不同频率")
print("=" * 60)

dim = 8
base = 10000.0

# 计算每个组的频率
freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
print(f"维度: {dim}, 分成 {dim//2} 组")
print(f"各组频率 θ: {freqs.tolist()}")
print(f"第0组频率最大 ({freqs[0]:.4f}) → 捕捉局部位置")
print(f"第{dim//2-1}组频率最小 ({freqs[-1]:.6f}) → 捕捉全局位置")

# 可视化：不同频率的旋转角度随位置的变化
print("\n位置  |  第0组角度  |  第3组角度")
print("-" * 40)
for pos in [0, 1, 5, 10, 50, 100]:
    angle_0 = pos * freqs[0].item()
    angle_3 = pos * freqs[3].item()
    print(f"  {pos:3d}  |  {angle_0:8.4f}   |  {angle_3:10.6f}")

print("\n→ 高频组：位置1和2的角度差很大，能区分相邻位置")
print("→ 低频组：位置1和2的角度差很小，但位置1和100的角度差明显")


# ============================================================
# 第四步：手动实现 RoPE
# ============================================================

def rope_precompute(dim, max_seq_len, base=10000.0):
    """预计算 RoPE 的 cos 和 sin 表

    Args:
        dim: 向量维度（head_dim）
        max_seq_len: 最大序列长度
        base: 频率基数（默认 10000）

    Returns:
        cos_table: [max_seq_len, dim]
        sin_table: [max_seq_len, dim]
    """
    # 每组的频率 θ_i = 1 / (base^(2i/d))
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # 位置索引 t = 0, 1, 2, ..., max_seq_len-1
    t = torch.arange(max_seq_len)

    # 角度矩阵：t_i * θ_j，形状 [max_seq_len, dim//2]
    angles = torch.outer(t, freqs)

    # 扩展到 dim 维：每组 cos/sin 重复一次
    # [cos(θ0), cos(θ0), cos(θ1), cos(θ1), ...]
    cos_table = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin_table = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)

    return cos_table, sin_table


def apply_rope(x, cos, sin):
    """对输入向量应用旋转位置编码

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    Returns:
        旋转后的向量，形状与 x 相同
    """
    cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim]
    sin = sin.unsqueeze(1)  # [seq_len, 1, head_dim]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]

    cos_h = cos[..., :half]
    sin_h = sin[..., :half]

    out = torch.cat([x1 * cos_h - x2 * sin_h,
                     x2 * cos_h + x1 * sin_h], dim=-1)
    return out


# 实验4：手动实现 RoPE
print("\n" + "=" * 60)
print("实验4：手动实现 RoPE")
print("=" * 60)

head_dim = 8
seq_len = 5
num_heads = 2
batch_size = 1

# 模拟 Q 向量
torch.manual_seed(42)
xq = torch.randn(batch_size, seq_len, num_heads, head_dim)

# 预计算 cos/sin
cos_table, sin_table = rope_precompute(head_dim, max_seq_len=10)

print(f"输入形状: {xq.shape}")
print(f"cos表形状: {cos_table.shape}")
print(f"sin表形状: {sin_table.shape}")

# 取前 seq_len 个位置的 cos/sin
cos = cos_table[:seq_len]
sin = sin_table[:seq_len]

# 应用 RoPE
xq_rope = apply_rope(xq, cos, sin)

print(f"\n位置0的原始向量 (head0): {xq[0, 0, 0].tolist()}")
print(f"位置0的旋转后向量 (head0): {xq_rope[0, 0, 0].tolist()}")
print(f"位置1的原始向量 (head0): {xq[0, 1, 0].tolist()}")
print(f"位置1的旋转后向量 (head0): {xq_rope[0, 1, 0].tolist()}")

# 验证：位置0不旋转（cos=1, sin=0）
print(f"\n位置0的cos值: {cos[0].tolist()}")
print(f"位置0的sin值: {sin[0].tolist()}")
print("位置0的cos全为1、sin全为0 → 位置0不旋转！")


# ============================================================
# 第五步：验证 RoPE 的相对位置性质
# ============================================================

print("\n" + "=" * 60)
print("实验5：RoPE 编码相对位置")
print("=" * 60)

head_dim = 16
seq_len = 20

# 两个固定的向量 q 和 k
torch.manual_seed(0)
q_vec = torch.randn(1, 1, 1, head_dim)
k_vec = torch.randn(1, 1, 1, head_dim)

cos_t, sin_t = rope_precompute(head_dim, max_seq_len=100)

# 计算 q 在位置 m 和 k 在位置 n 的内积
print("相对位置差 | 内积值")
print("-" * 30)
for m in [5]:
    q_rot = apply_rope(q_vec, cos_t[m:m+1], sin_t[m:m+1])
    for n_offset in [0, 1, 2, 3, 5, 10, 20, 50]:
        n = m + n_offset
        k_rot = apply_rope(k_vec, cos_t[n:n+1], sin_t[n:n+1])
        dot = torch.dot(q_rot[0, 0, 0], k_rot[0, 0, 0]).item()
        print(f"   {n_offset:2d}       |  {dot:.4f}")

print("\n→ 相对位置差越大，内积越小（远距离衰减）")
print("→ 这是 RoPE 的核心优势：注意力天然偏向近处")


# ============================================================
# 第六步：与其他位置编码对比
# ============================================================

print("\n" + "=" * 60)
print("实验6：位置编码方法对比")
print("=" * 60)

methods = {
    "绝对位置编码 (原始Transformer)": "给每个位置学一个向量，加到输入上\n  缺点：不能外推到训练时没见过的长度",
    "ALiBi": "在注意力分数上加线性偏置\n  优点：天然支持长度外推\n  缺点：没有绝对位置信息",
    "RoPE (MiniMind使用)": "用旋转编码位置\n  优点：编码相对位置 + 远距离衰减 + 无额外参数\n  缺点：长度外推需要额外技巧（如YaRN）",
}

for name, desc in methods.items():
    print(f"\n{name}:")
    print(f"  {desc}")


# ============================================================
# 第七步：MiniMind 中的 RoPE 实现
# ============================================================

print("\n" + "=" * 60)
print("实验7：MiniMind 中的 RoPE 实现")
print("=" * 60)

# MiniMind 的 precompute_freqs_cis 实现
def precompute_freqs_cis(dim, end=32768, rope_base=1e6):
    """MiniMind 的 RoPE 预计算（简化版，不含 YaRN scaling）"""
    # 频率：θ_i = 1 / (base^(2i/d))
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 位置索引
    t = torch.arange(end)
    # 角度矩阵
    freqs = torch.outer(t, freqs).float()
    # cos/sin 表，每个值重复一次以匹配 dim 维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

# MiniMind 的 apply_rotary_pos_emb 实现
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """MiniMind 的 RoPE 应用函数"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    cos_h, sin_h = cos[..., :half], sin[..., :half]
    q_out = torch.cat([q1 * cos_h - q2 * sin_h, q2 * cos_h + q1 * sin_h], dim=-1)
    k_out = torch.cat([k1 * cos_h - k2 * sin_h, k2 * cos_h + k1 * sin_h], dim=-1)
    return q_out, k_out

# 模拟 MiniMind 的使用方式
head_dim = 64
num_heads = 8
num_kv_heads = 4
seq_len = 16
batch_size = 2
hidden_size = num_heads * head_dim

# 预计算（在模型初始化时做一次）
freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=2048)
print(f"预计算 cos 表形状: {freqs_cos.shape}")
print(f"预计算 sin 表形状: {freqs_sin.shape}")
print(f"一次预计算，所有位置共享！")

# 模拟 Attention 中的使用
xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
xk = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)

cos = freqs_cos[:seq_len]
sin = freqs_sin[:seq_len]

xq_rope, xk_rope = apply_rotary_pos_emb(xq, xk, cos, sin)

print(f"\nQ 形状: {xq.shape} → 旋转后: {xq_rope.shape}")
print(f"K 形状: {xk.shape} → 旋转后: {xk_rope.shape}")
print(f"RoPE 不改变向量形状，只改变方向！")

# 验证：旋转后向量的模长不变
norm_before = xq[0, 0, 0].norm().item()
norm_after = xq_rope[0, 0, 0].norm().item()
print(f"\n旋转前向量模长: {norm_before:.4f}")
print(f"旋转后向量模长: {norm_after:.4f}")
print(f"模长不变: {abs(norm_before - norm_after) < 0.01}")
print("→ 旋转只改变方向，不改变大小")


# ============================================================
# 第八步：RoPE 的 base 参数与频率范围
# ============================================================

print("\n" + "=" * 60)
print("实验8：base 参数的影响")
print("=" * 60)

for base in [100.0, 10000.0, 1000000.0]:
    freqs = 1.0 / (base ** (torch.arange(0, 16, 2).float() / 16))
    print(f"\nbase = {base:.0f}:")
    print(f"  最高频率: {freqs[0]:.4f} (旋转最快，区分相邻位置)")
    print(f"  最低频率: {freqs[-1]:.6f} (旋转最慢，区分远距离)")
    # 最低频率转一圈需要多少个位置
    period = 2 * math.pi / freqs[-1].item()
    print(f"  最低频率周期: {period:.0f} 个位置")

print("\nMiniMind 使用 base = 1,000,000（比默认的 10,000 大100倍）")
print("→ 更大的 base → 更低的最低频率 → 更长的周期")
print("→ 适合处理更长的序列（支持更远距离的位置区分）")


# ============================================================
# 第九步：RoPE 在 Attention 中的完整流程
# ============================================================

print("\n" + "=" * 60)
print("实验9：RoPE 在 Attention 中的完整流程")
print("=" * 60)

print("""
Attention 中 RoPE 的使用流程：

1. 输入 x: [batch, seq_len, hidden_size]
2. 线性投影得到 Q, K, V:
   xq = q_proj(x)  → [batch, seq_len, num_heads, head_dim]
   xk = k_proj(x)  → [batch, seq_len, num_kv_heads, head_dim]
   xv = v_proj(x)  → [batch, seq_len, num_kv_heads, head_dim]

3. 对 Q 和 K 应用 RoPE（V 不需要！）:
   xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

4. 计算注意力分数:
   scores = xq @ xk^T / sqrt(head_dim)
   → 由于 RoPE，分数自动包含相对位置信息！

5. 注意力加权求和:
   output = softmax(scores) @ xv

关键点：
  - 只对 Q 和 K 旋转，V 不旋转
  - cos/sin 在模型初始化时预计算，不需要学习
  - RoPE 不增加任何可训练参数
""")


# ============================================================
# 第十步：RoPE 的直觉图示
# ============================================================
print("\n" + "=" * 60)
print("实验10：旋转的直观图示")
print("=" * 60)

print("""
想象你站在时钟的中心, 把每个词的向量看成一根指针:

       0°
       │
   ────┼────
   │       │
180°  ●   0°  
   │       │
   ────┼────
       │
      90°

位置 0: 指针指向某个方向 (比如 12 点钟 = 0°)
位置 1: 指针顺时针旋转 θ 角度
位置 2: 指针顺时针旋转 2θ 角度
位置 m: 指针顺时针旋转 mθ 角度

不同维度的"指针"用不同速度旋转:
  维度 0-1:  转得快 (高频) → 区分 "我" 和 "我" 后面紧跟的字
  维度 2-3:  转得慢 (低频) → 区分 "我" 和段落开头

→ 高频维度负责"近距离关系" (局部)
→ 低频维度负责"远距离关系" (全局)

这就像卫星导航:
  定位时同时用 GPS (高频, 精确) 和 罗盘 (低频, 大方向)
  两者结合才能精确定位
""")


# ============================================================
# 第十一步：RoPE 的数学性质
# ============================================================
print("\n" + "=" * 60)
print("实验11：RoPE 的三大数学性质")
print("=" * 60)

print("""
性质1: 相对位置编码
─────────────────
  <R(mθ)·q, R(nθ)·k> = <q, k> · cos((m-n)θ)

  含义: 两个旋转后向量的内积只取决于相对距离 (m-n)
  例: q 在位置 5, k 在位置 8
      距离 = 3
      和 q 在位置 10, k 在位置 13 的结果一样
      → 模型天然能理解"距离3"这个概念

性质2: 远距离衰减
─────────────────
  当 |m-n| 很大时, cos((m-n)θ) 接近 0 (甚至震荡)
  → 距离越远的两个位置, 注意力分数越低
  → 模型自动更关注邻近位置

  注意: 当距离非常大时, cos 会震荡 (在 +1 和 -1 之间)
  这就是为什么 RoPE 难以外推到极长序列
  (可以用 YaRN / NTK-aware 插值解决)

性质3: 长度无关
─────────────────
  cos/sin 表只和 head_dim 有关
  训练时用 2048 长度, 推理时可以用 4096
  只要 cos/sin 表能索引到对应位置就行
  → 天然支持"训练短, 推长"
""")


# ============================================================
# 第十二步：RoPE vs 传统位置编码对比
# ============================================================
print("\n" + "=" * 60)
print("实验12：RoPE vs 传统位置编码对比")
print("=" * 60)

print("""
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ 方案              │ 位置信息     │ 相对位置     │ 长度外推     │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 绝对位置 (BERT)   │ ✓ 加到输入   │ ✗ 难学到     │ ✗ 不可外推   │
│ 相对位置偏置 (T5) │ ✗ 无绝对     │ ✓ 加到分数   │ ✓ 可外推     │
│ ALiBi             │ ✗ 无绝对     │ ✓ 线性偏置   │ ✓ 可外推     │
│ RoPE              │ ✓ 隐含绝对   │ ✓ 内积自带   │ △ 需要技巧   │
└──────────────────┴──────────────┴──────────────┴──────────────┘

RoPE 的独特之处:
  - 既有绝对位置 (旋转 mθ, 包含 m 的信息)
  - 又有相对位置 (内积只依赖 m-n)
  - 还可以远距离衰减 (cos 的性质)
  → "三位一体", 非常优雅!
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】为什么 V 不需要旋转?
  RoPE 只对 Q 和 K 应用旋转, 但 V 不旋转。为什么?

【练习2】相对位置编码
  假设 q 在位置 3, k 在位置 7
  相对距离是多少? 与 q 在位置 10, k 在位置 14 的距离一样吗?

【练习3】base 参数选择
  base 越大, 最低频率越低, 周期越长。
  如果训练序列长度 2048, 推理想外推到 16384, base 应该调大还是调小?

【练习4】维度配对
  RoPE 将 d 维向量分成 d/2 组, 每组 2 维
  为什么是 2 维一组, 而不是 4 维或 8 维?

【练习5】RoPE 与 KV Cache
  使用 KV Cache 推理时, 每生成一个新 token, 它在位置 m
  Q 需要应用位置 m 的旋转, K 已经缓存了
  K 缓存时是哪个位置的旋转?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  V 表示'这个位置有什么信息', 不需要位置信息")
print("  Q 和 K 用来计算'哪些位置相关', 需要位置信息")
print("  注意力公式: softmax(QK^T)V")
print("  Q 和 K 决定权重, V 提供内容")
print("  → V 不旋转, 内容的语义不变, 只改变注意力分布")

# 练习2
print("\n【练习2 答案】")
print("  q 在位置 3, k 在位置 7: 相对距离 = 7 - 3 = 4")
print("  q 在位置 10, k 在位置 14: 相对距离 = 14 - 10 = 4")
print("  → 相对距离一样, 内积一样!")
print("  这就是 RoPE 的相对位置性质: <R(mθ)q, R(nθ)k> 只依赖 m-n")

# 演示
def test_relative():
    head_dim = 16
    cos_t, sin_t = rope_precompute(head_dim, max_seq_len=100)
    torch.manual_seed(42)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    for m, n in [(3, 7), (10, 14), (0, 4)]:
        q_rot = apply_rope(q, cos_t[m:m+1], sin_t[m:m+1])
        k_rot = apply_rope(k, cos_t[n:n+1], sin_t[n:n+1])
        dot = torch.dot(q_rot[0, 0, 0], k_rot[0, 0, 0]).item()
        print(f"    位置 m={m}, n={n}, 距离={n-m}, 内积={dot:.4f}")
test_relative()

# 练习3
print("\n【练习3 答案】")
print("  base 应该调大 (例如从 10000 调到 100000)")
print("  训练 2048 用 base=10000:")
print("    最低频率周期 = 2π / 最低频率 ≈ 2048 位置")
print("  推理 16384, 想覆盖同样范围:")
print("    需要最低频率周期 ≈ 16384 位置")
print("    → 最低频率要除以 8")
print("    → base 要乘以 8")
print("  → 这种方法叫'NTK-aware scaling'")
print("  更复杂的方法是 YaRN, 见第16课")

# 演示
for base in [10000.0, 100000.0]:
    freqs = 1.0 / (base ** (torch.arange(0, 16, 2).float() / 16))
    period = 2 * math.pi / freqs[-1].item()
    print(f"    base={base:.0f}: 最低频率周期 = {period:.0f} 位置")

# 练习4
print("\n【练习4 答案】")
print("  2 维一组, 因为 2D 平面上的旋转是基础操作")
print("  旋转矩阵 R(θ) = [cos θ, -sin θ; sin θ, cos θ] 需要 2D")
print("  如果是 4 维一组, 可以分解成两个 2D 旋转的组合")
print("  → 2 维一组是最简单、最高效的方案")
print("  → 矩阵形式简单, 数值稳定, 易于 GPU 并行")
print()
print("  RoPE 的实现中:")
print("    [x0, x1] → [x0*cos - x1*sin, x1*cos + x0*sin]")
print("  只用 4 次乘法和 2 次加法, 非常高效!")

# 练习5
print("\n【练习5 答案】")
print("  K 缓存的是它生成时所在位置的旋转")
print("  例: 推理时已经处理了位置 0, 1, 2 的 token")
print("      它们的 K 分别旋转了 0θ, 1θ, 2θ 后存入缓存")
print()
print("  生成新 token (位置 3) 时:")
print("    Q 用 3θ 旋转")
print("    K 用缓存 (已经是 0θ, 1θ, 2θ 旋转)")
print("    → 计算内积时, 自动得到相对距离 3, 2, 1, 0")
print()
print("  KV Cache 配合 RoPE 的关键:")
print("    K 不能重新旋转! 否则相对位置会乱")
print("    只需给新的 Q 加上对应位置的旋转即可")


# ============================================================
# 本课小结
# ============================================================
print("=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. 位置编码让模型区分语序（"猫吃鱼" vs "鱼吃猫"）
2. RoPE 用旋转编码位置：位置 m 的向量旋转 mθ
3. 内积只依赖相对位置差，天然编码相对位置
4. 高维向量分组旋转，不同组用不同频率
5. MiniMind 预计算 cos/sin 表，运行时直接查表
6. 只对 Q 和 K 旋转，V 不旋转，不增加参数

数据流至此：
  Token IDs → Embedding → RMSNorm → RoPE旋转 → ...
  [42,108]   [0.1,-0.3]  [0.8,0.2]  旋转后的Q/K

下一步 → lesson05_attention.py：核心机制——词与词如何相互关注？
""")
