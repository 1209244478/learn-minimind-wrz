"""
第3课：RMSNorm — 为什么需要归一化？
=====================================

深层网络的"训练难题":
  随着网络层数加深, 激活值的分布会"漂移"
  - 有些层输出爆炸 (数值极大, e.g. 1e6)
  - 有些层输出消失 (数值极小, e.g. 1e-6)
  - 导致梯度不稳定, 训练失败

归一化层 (Normalization) 的作用:
  把每层输出"拉回"到合理范围 (类似标准化)
  解决内部协变量偏移 (Internal Covariate Shift)
  让训练深层网络成为可能

类比:
  想象一个长跑接力赛, 每个人跑得越来越快或越来越慢
  归一化 = 每跑一段就"重新校准"速度
  保证每个人速度差不多, 接力稳定

LayerNorm vs BatchNorm vs RMSNorm:

  BatchNorm:  对 batch 维度归一化, 不适合变长序列
  LayerNorm:  对单个样本的所有特征归一化, 主流方案
  RMSNorm:    LayerNorm 的简化版, 计算更快, 效果相当

  公式对比:
    LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    RMSNorm:    x / sqrt(mean(x^2) + eps) * gamma

  RMSNorm 优势:
    ✓ 不需要减均值
    ✓ 不需要加偏置
    ✓ 计算量减少 ~30%
    ✓ 效果与 LayerNorm 相当

运行: python lessons/lesson03_rmsnorm.py
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 第1部分：为什么需要归一化
# ============================================================
print("=" * 60)
print("第1部分：为什么需要归一化")
print("=" * 60)


print("""
问题演示: 假设一个 50 层网络, 每层都做 y = W·x

  如果 W 的元素 = 0.99:
    50 层后: y = 0.99^50 * x ≈ 0.61 * x   (信号衰减)
    梯度也衰减 50 次, 接近 0
    → 梯度消失!

  如果 W 的元素 = 1.01:
    50 层后: y = 1.01^50 * x ≈ 1.64 * x   (信号放大)
    50 层后爆炸
    → 梯度爆炸!

  实际训练中, W 是学出来的, 可能某些层 W 偏大, 某些层偏小
  → 训练不稳定

归一化的核心作用:
  ✓ 稳定每层输出的分布
  ✓ 让梯度大小适中
  ✓ 加速训练收敛
  ✓ 允许更大学习率
""")


# 演示数值爆炸/消失
print("\n[演示] 深层网络中的数值问题")
print("-" * 60)
print(f"{'层数':<8}{'W=0.99':<15}{'W=1.01':<15}{'W=1.5':<15}{'W=2.0':<15}")
print("-" * 60)
for depth in [10, 30, 50, 100]:
    v_099 = 0.99 ** depth
    v_101 = 1.01 ** depth
    v_15 = 1.5 ** depth
    v_20 = 2.0 ** depth
    print(f"{depth:<8}{v_099:<15.4f}{v_101:<15.4f}{v_15:<15.2e}{v_20:<15.2e}")

print("\n观察: W=1.5 在 50 层后已经到 6e8, 数值爆炸!")


# ============================================================
# 第2部分：LayerNorm 实现
# ============================================================
print("\n" + "=" * 60)
print("第2部分：LayerNorm 原理与实现")
print("=" * 60)


class LayerNorm(nn.Module):
    """LayerNorm - 对单个样本的特征维度归一化"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))   # 缩放
        self.beta = nn.Parameter(torch.zeros(dim))   # 偏移

    def forward(self, x):
        # x: [..., dim]
        mean = x.mean(dim=-1, keepdim=True)          # 最后一个维度的均值
        var = x.var(dim=-1, keepdim=True)            # 方差
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# 演示
ln = LayerNorm(8)
x = torch.randn(2, 4, 8) * 5 + 10  # 任意分布
y = ln(x)

print(f"\n输入 x 统计: mean={x.mean():.2f}, std={x.std():.2f}")
print(f"输出 y 统计: mean={y.mean():.2f}, std={y.std():.2f}")
print(f"\n输出形状: {y.shape}")
print(f"gamma shape: {ln.gamma.shape}, beta shape: {ln.beta.shape}")


# ============================================================
# 第3部分：RMSNorm 实现
# ============================================================
print("\n" + "=" * 60)
print("第3部分：RMSNorm 原理与实现")
print("=" * 60)


class RMSNorm(nn.Module):
    """RMSNorm - 只用均方根, 不用均值"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x: [..., dim]
        # 均方根: sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.gamma * self._norm(x)


# 演示
rms = RMSNorm(8)
y = rms(x)
print(f"\n输入 x 统计: mean={x.mean():.2f}, std={x.std():.2f}")
print(f"输出 y 统计: mean={y.mean():.2f}, std={y.std():.2f}")
print(f"\n观察: y 范围比 x 小很多, 但不严格是 0 均值 (RMSNorm 特性)")


# ============================================================
# 第4部分：RMSNorm vs LayerNorm 对比
# ============================================================
print("\n" + "=" * 60)
print("第4部分：RMSNorm vs LayerNorm 对比")
print("=" * 60)

# 性能对比
import time

dim = 4096
batch = 32
seq = 2048
x = torch.randn(batch, seq, dim)

# 预热
for _ in range(5):
    _ = LayerNorm(dim)(x)
    _ = RMSNorm(dim)(x)

# LayerNorm 计时
start = time.time()
for _ in range(50):
    _ = LayerNorm(dim)(x)
ln_time = (time.time() - start) / 50 * 1000

# RMSNorm 计时
start = time.time()
for _ in range(50):
    _ = RMSNorm(dim)(x)
rms_time = (time.time() - start) / 50 * 1000

print(f"\n性能对比 (dim={dim}, batch={batch}, seq={seq}):")
print(f"  LayerNorm: {ln_time:.3f} ms/次")
print(f"  RMSNorm:   {rms_time:.3f} ms/次")
print(f"  加速:      {ln_time/rms_time:.2f}x")

# 计算量对比
print("""
\n计算量分析:
  LayerNorm: 计算 mean + var + 减均值 + 除法 + 乘gamma + 加beta
  RMSNorm:   计算 mean(x²) + 除法 + 乘gamma

  RMSNorm 少了:
    - 一次减法 (减均值)
    - 一次加法 (加偏置)
    - 一次平方根
  → 减少 ~30-40% 计算量
""")


# ============================================================
# 第5部分：归一化的位置
# ============================================================
print("\n" + "=" * 60)
print("第5部分：归一化放在哪里？")
print("=" * 60)

print("""
归一化有 2 种主要位置:

1) Post-LN (传统 Transformer)
   x → Attention → Add → LN → FFN → Add → LN
   
   问题: 残差流的激活值会越来越大
   现代模型已少用

2) Pre-LN (现代主流, MiniMind 用这种)
   x → LN → Attention → Add → LN → FFN → Add
   
   优点: 残差流稳定, 训练更容易
   缺点: 注意力层的激活可能不一致

可视化:
  Post-LN:
    [x] → [Attn] → [+] → [LN] → [FFN] → [+] → [LN] → 输出
              ↓                ↓
              └──残差──────────┘
    
  Pre-LN (推荐):
    [x] → [LN] → [Attn] → [+] → [LN] → [FFN] → [+] → 输出
            ↓                ↓        ↓
            └──残差──────────┘        │
                     └──残差──────────┘
""")


# ============================================================
# 第6部分：RMSNorm 在 MiniMind 中的应用
# ============================================================
print("\n" + "=" * 60)
print("第6部分：RMSNorm 在 MiniMind 中")
print("=" * 60)

print("""
MiniMind 的 RMSNorm 用在 3 个地方:

1) 输入层:
   x = RMSNorm(embed(x))    # 嵌入后立即归一化

2) 每个 Transformer Block (2 次):
   x = x + Attn(RMSNorm(x))     # 注意力前
   x = x + FFN(RMSNorm(x))      # FFN 前

3) 最终输出:
   x = RMSNorm(x)            # LM Head 前再归一化一次

总参数量影响:
   RMSNorm 每层: d 个 gamma 参数
   MiniMind 8 层, hidden=512: 8 × 512 = 4096 参数
   占比: < 0.1% (几乎无开销)

实现 (PyTorch):
```python
class MiniMindRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
```
""")


# ============================================================
# 第7部分：归一化变体总结
# ============================================================
print("\n" + "=" * 60)
print("第7部分：归一化变体总结")
print("=" * 60)

comparison = """
┌────────────┬──────────┬──────────┬──────────┬──────────┐
│ 方案        │ 归一化维度│ 计算量   │ 适用场景  │ 代表模型  │
├────────────┼──────────┼──────────┼──────────┼──────────┤
│ BatchNorm  │ batch    │ 中       │ CNN      │ ResNet   │
│ LayerNorm  │ 单样本   │ 中       │ 通用     │ BERT     │
│ RMSNorm    │ 单样本   │ 低       │ LLM     │ LLaMA    │
│ DeepNorm   │ 单样本   │ 中       │ 极深模型 │ GPT-3    │
│ GroupNorm  │ channel组│ 中       │ 小batch │ ConvNeXt │
└────────────┴──────────┴──────────┴──────────┴──────────┘

为什么 LLM 都用 RMSNorm?
  - LLM 序列长度变化大, BatchNorm 不可用
  - 比 LayerNorm 更快, 效果相当
  - 训练时更稳定 (数值范围更小)
"""

print(comparison)


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】数值爆炸分析
  假设 W = 1.5 的矩阵乘法, 多少层后输出会超过 1e6?

【练习2】LayerNorm 输出范围
  LayerNorm 处理后的数据, mean 和 std 大约是多少?

【练习3】RMSNorm 的均值
  RMSNorm 不会让数据严格 0 均值, 为什么?

【练习4】Pre-LN vs Post-LN
  为什么现代 LLM (LLaMA, GPT) 都用 Pre-LN?

【练习5】gamma 参数的作用
  RMSNorm 中有可学习参数 gamma (默认初始化为 1)
  训练后 gamma 会变吗? 大概是什么值?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  1.5^n = 1e6")
print("  n × log(1.5) = log(1e6)")
print("  n = 6 / log10(1.5) = 6 / 0.176 = 34.1")
print("  → 大约 35 层后数值爆炸到 1e6")

# 实际验证
import math
print("\n  验证:")
for n in [10, 20, 30, 35, 40]:
    val = 1.5 ** n
    print(f"    1.5^{n} = {val:.2e}")

# 练习2
print("\n【练习2 答案】")
print("  LayerNorm 后: mean ≈ 0, std ≈ 1")
print("  gamma 和 beta 不变时, 输出严格 mean=0, std=1")
print("  训练后 gamma 和 beta 会调整输出分布")

# 实际验证
ln = LayerNorm(8)
x = torch.randn(100, 8) * 10 + 5
y = ln(x)
print(f"\n  验证: y mean={y.mean():.4f}, y std={y.std():.4f}")

# 练习3
print("\n【练习3 答案】")
print("  RMSNorm 公式: x / sqrt(mean(x²) + eps) * gamma")
print("  没有减均值步骤, 所以输出不一定 0 均值")
print("  例: 输入 [1, 2, 3, 4]")
print("  RMS = sqrt(mean([1,4,9,16])) = sqrt(7.5) ≈ 2.739")
print("  缩放后 ≈ [0.365, 0.730, 1.095, 1.461]")
print("  均值 ≈ 0.913, 不是 0")

# 演示
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
rms = RMSNorm(4)
y = rms(x)
print(f"\n  验证: 输入 {x.tolist()[0]}")
print(f"        输出 {[round(v, 3) for v in y.tolist()[0]]}")
print(f"        均值 = {y.mean():.3f}")

# 练习4
print("\n【练习4 答案】")
print("  Pre-LN 优势:")
print("  1. 残差流的方差在层间保持稳定 (不会指数增长)")
print("  2. 训练时梯度直接通过残差流回传, 不经过 LN")
print("  3. 深层 (100+) 模型训练更容易")
print("  Post-LN 缺点:")
print("  1. 残差路径上的激活值会随深度指数增长")
print("  2. 需要 warmup 学习率, 训练不稳定")

# 练习5
print("\n【练习5 答案】")
print("  会变! gamma 初始化为 1, 训练后会调整")
print("  gamma 的值反映该维度的重要性")
print("  - 重要维度: gamma 较大 (放大)")
print("  - 不重要维度: gamma 较小 (缩小)")
print("  - 可被归一化的维度: gamma ≈ 0")
print("  训练后, 多数 gamma 在 0.5-2.0 范围")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. 为什么需要归一化
   - 深层网络有数值爆炸/消失问题
   - 归一化让训练稳定
   - 允许更大学习率

2. LayerNorm
   - 公式: (x - mean) / sqrt(var) * gamma + beta
   - 对单个样本的特征维度归一化
   - 通用方案

3. RMSNorm (MiniMind 用)
   - 公式: x / sqrt(mean(x²)) * gamma
   - 比 LayerNorm 简单 30%+
   - 效果相当, 速度更快
   - 现代 LLM 的选择

4. 位置选择
   - Pre-LN (LN 在子层前) 是主流
   - 比 Post-LN 训练更稳定
   - MiniMind 也用 Pre-LN

5. 实现简单
   - 只有 1 个参数 (gamma)
   - 几乎不增加参数量
   - 推理开销很小

下一步: 第4课 - RoPE (模型如何知道词的位置？)
""")
