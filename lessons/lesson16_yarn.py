"""
第16课：YaRN — 让模型学会"举一反三"的长度外推
================================================

问题: 模型在 2048 长度训练，能用在 8192 长度吗？
  标准 RoPE: 不能！性能急剧下降
  原因: 训练时只见过小位置，推理时遇到大位置

什么是"长度外推"？
  训练序列长度: 2048
  推理序列长度: 8192 (超出训练长度的 4x)

  训练:  在 1-2048 的位置上见过
  推理:  突然出现 5000 位置, 模型完全没学过

YaRN (Yet another RoPE extensioN) 解决方案:
  1. 缩放位置: pos' = pos / s
  2. 注意力分数修正: 补偿熵的变化
  3. 大部分频率按比例缩放
  4. 训练短，推理长也能用

运行: python lessons/lesson16_yarn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 第1部分：长度外推问题
# ============================================================

print("=" * 60)
print("第1部分：长度外推问题")
print("=" * 60)

print("""
假设模型在 2048 长度训练:

训练时:
  - 位置 1 出现词: "今天"
  - 位置 100 出现词: "天气"
  - 位置 500 出现词: "很好"
  - 位置 2000 出现词: "啊"

推理时，喂入 4096 长度文本:
  - 位置 1-2048: 模型见过, 表现正常
  - 位置 2049-4096: 模型完全没见过！
  - 表现: 乱码、性能崩溃

为什么会这样？
  原因1: 位置编码超出训练范围
  原因2: 注意力分布与训练时完全不同
  原因3: 大数值的正弦/余弦值, 模型没学到
""")

# 演示 RoPE 在不同长度的差异
def rope_freqs(head_dim, position, base=10000):
    """计算 RoPE 频率"""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = position * freqs
    return torch.cos(angles), torch.sin(angles)

print("\n[演示] RoPE 在不同位置的三角函数值")
print("-" * 60)
print("位置     | cos(θ) (低频分量) | cos(θ) (高频分量)")
print("-" * 60)
head_dim = 64
for pos in [100, 1000, 2048, 4096, 8192]:
    cos_vals, _ = rope_freqs(head_dim, torch.tensor([pos]))
    print(f"位置 {pos:5d} | {cos_vals[0].item():+.4f}            | {cos_vals[-1].item():+.4f}")

print("\n观察: 高频分量在不同位置变化剧烈")
print("训练时位置 < 2048, 模型只学到了小范围的变化模式")


# ============================================================
# 第2部分：位置插值 (Position Interpolation)
# ============================================================

print("\n" + "=" * 60)
print("第2部分：位置插值 (PI)")
print("=" * 60)

print("""
最简单的方案: 把所有位置"压缩"到训练范围内

原始: position = 1, 2, ..., 4096 (外推)
插值: position' = position / 4    (缩放 4x)
      = 0.25, 0.5, ..., 1024     (在训练范围内)

公式:  pos' = pos × (L_train / L_inference)

优点:
  ✓ 简单, 一行代码搞定
  ✓ 保留训练时学到的位置信息

缺点:
  ✗ 位置"分辨率"降低, 远处的位置区分度差
  ✗ 还需要微调才能恢复性能
""")

# 演示位置插值
print("\n[演示] 位置插值 (scale_factor=4)")
print("-" * 60)
print("原始位置  | 缩放后位置  | 映射范围")
print("-" * 60)
scale = 4
train_len = 2048
for pos in [100, 1000, 2048, 4096, 8192]:
    scaled = pos / scale
    print(f"{pos:8d} | {scaled:8.1f}   | {'训练范围内' if scaled <= train_len else '超出训练'}")


# ============================================================
# 第3部分：YaRN — 精细的位置外推
# ============================================================

print("\n" + "=" * 60)
print("第3部分：YaRN — 精细的位置外推")
print("=" * 60)

print("""
YaRN 的核心洞察:
  RoPE 频率有高有低
  - 低频维度: 编码粗粒度位置 (范围大)
  - 高频维度: 编码细粒度位置 (范围小)

  简单缩放对高频维度不利 (会丢失细节)
  简单缩放对低频维度合适

YaRN 的处理:
  1. 把 RoPE 频率分成3组:
     - 高频: 不缩放 (保留细节)
     - 中频: 平滑过渡
     - 低频: 完整缩放

  2. 对每个维度 d 计算:
     h(d) = 缩放因子 (与维度有关)

  3. 缩放位置: pos' = pos × h(d)

  4. 注意力温度修正:
     t' = 0.1 × ln(s) + 1    (s 是缩放因子)
     attn' = attn / t'

效果:
  ✓ 比简单插值效果好
  ✓ 不需要或只需少量微调
  ✓ 4x-16x 长度外推都能用
""")


# ============================================================
# 第4部分：YaRN 完整实现
# ============================================================

print("\n" + "=" * 60)
print("第4部分：YaRN 完整实现")
print("=" * 60)


def get_yarn_mscale(scale_factor):
    """计算 YaRN 的 mscale (温度修正)"""
    if scale_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scale_factor) + 1.0


def precompute_yarn_freqs_cis(dim, end, original_max=2048, scale_factor=4.0,
                               beta_fast=32, beta_slow=1):
    """YaRN 频率预计算"""
    # 基础频率
    freqs_base = 1.0 / (10000 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 找边界频率
    if scale_factor <= 1.0:
        # 不缩放, 返回完整 RoPE 频率
        t = torch.arange(end)
        freqs = torch.outer(t, freqs_base).float()
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        return freqs_cos, freqs_sin

    # 找到 fast 和 slow 的频率边界
    inv_freq_extrapolation = 1.0 / (scale_factor * 10000 ** (
        torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    inv_freq_interpolation = 1.0 / (10000 ** (
        torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 计算每个维度的缩放因子
    low_freq_factor = scale_factor
    high_freq_factor = 1.0
    freq_wavelen = 2 * math.pi / inv_freq_interpolation

    # 边界波长
    low_freq_wavelen = original_max / low_freq_factor
    high_freq_wavelen = original_max / high_freq_factor

    # 计算每个维度的缩放
    smooth = (freq_wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    smooth = smooth.clamp(0, 1)
    inv_freq = (1 - smooth) * inv_freq_extrapolation + smooth * inv_freq_interpolation

    # 【频率分组的直觉解释】
    #
    # YaRN 把频率分成三组，每组用不同策略：
    #
    # 高频组（波长 < high_freq_wavelen）：
    #   smooth ≈ 0 → inv_freq ≈ inv_freq_extrapolation（直接外推）
    #   为什么？高频已经能区分相邻位置，不需要插值
    #   类比：秒针不需要调慢，1秒就是1秒
    #
    # 低频组（波长 > low_freq_wavelen）：
    #   smooth ≈ 1 → inv_freq ≈ inv_freq_interpolation（线性插值）
    #   为什么？低频在长序列上会"转太多圈"，需要压缩
    #   类比：时针在长周期上需要调整刻度
    #
    # 中间组（high_freq < 波长 < low_freq）：
    #   smooth 在 0~1 之间 → 混合外推和插值
    #   为什么？这些频率既需要保持区分力，又不能转太快
    #   类比：分针需要适度调整
    #
    # 数值示例（scale_factor=4, original_max=2048）：
    #   high_freq_wavelen = 2048 / 1.0 = 2048
    #   low_freq_wavelen  = 2048 / 4.0 = 512
    #   波长 < 512 → 纯外推（高频，保持原样）
    #   波长 > 2048 → 纯插值（低频，压缩频率）
    #   512 < 波长 < 2048 → 平滑过渡

    # 位置 × 频率
    t = torch.arange(end)
    freqs = torch.outer(t, inv_freq).float()

    # 计算 mscale
    mscale = get_yarn_mscale(scale_factor)

    freqs_cos = torch.cat([torch.cos(freqs) * mscale, torch.cos(freqs) * mscale], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs) * mscale, torch.sin(freqs) * mscale], dim=-1)
    return freqs_cos, freqs_sin


# 演示 YaRN 频率
print("\n[演示] YaRN 频率 vs 原始 RoPE 频率")
print("-" * 60)
print("位置 5000 的频率值:")
print(f"{'维度':<10}{'原始 RoPE':<20}{'YaRN (4x)':<20}")
print("-" * 60)
cos_orig, _ = precompute_yarn_freqs_cis(32, 5001, scale_factor=1.0)
cos_yarn, _ = precompute_yarn_freqs_cis(32, 5001, scale_factor=4.0)

for d in [0, 8, 16, 24]:
    orig_val = cos_orig[5000, d].item()
    yarn_val = cos_yarn[5000, d].item()
    print(f"{d:<10}{orig_val:<20.4f}{yarn_val:<20.4f}")


# ============================================================
# 第5部分：应用 YaRN
# ============================================================

print("\n" + "=" * 60)
print("第5部分：应用 YaRN 到 Attention")
print("=" * 60)

print("""
使用 YaRN 只需要替换 RoPE 频率计算:

```python
# 原始 RoPE
freqs_cos, freqs_sin = precompute_freqs_cis(dim, end)

# YaRN
freqs_cos, freqs_sin = precompute_yarn_freqs_cis(
    dim, end,
    original_max=2048,    # 训练时的最大长度
    scale_factor=8.0,     # 缩放因子 (8192/2048=4, 但常用 8-16)
)
```

Attention 计算不变, 只是位置编码变了。
""")

# 演示：从 2048 外推到 8192
print("\n[实验] 验证 YaRN 支持的长度外推")
print("-" * 60)
print(f"训练长度: 2048, 目标长度: 8192, 缩放因子: 4x")

mscale = get_yarn_mscale(4.0)
print(f"\nYaRN mscale = {mscale:.4f}")
print(f"含义: 注意力分数需要除以 {mscale:.4f} 进行温度修正")
print(f"      (降低注意力分布的尖锐程度, 让模型更'关注长程')")


# ============================================================
# 第6部分：对比与选择
# ============================================================

print("\n" + "=" * 60)
print("第6部分：长度外推方案对比")
print("=" * 60)

comparison = """
┌──────────────┬──────────┬──────────┬──────────────────┐
│ 方案          │ 实现难度  │ 效果     │ 需要的微调        │
├──────────────┼──────────┼──────────┼──────────────────┤
│ 无 (直接外推) │ 0        │ 差       │ 0                │
│ 简单插值 (PI) │ 1        │ 一般     │ 100% 长度微调     │
│ NTK-aware    │ 3        │ 较好     │ 10% 长度微调      │
│ YaRN         │ 5        │ 优秀     │ 1% 长度微调       │
│ ABF          │ 5        │ 优秀     │ 10% 长度微调      │
└──────────────┴──────────┴──────────┴──────────────────┘

推荐:
  - 简单需求: 线性插值 + 全量长度微调
  - 平衡:     YaRN + 1% 微调
  - 极限:     YaRN + 关键样本微调
"""

print(comparison)


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. 长度外推问题
   - 模型训练长度 2048, 推理用 4096 就崩
   - 根本原因: 大位置没在训练中见过

2. 位置插值 (PI)
   - 把所有位置压缩到训练范围内
   - 实现简单, 但需要微调

3. YaRN 改进
   - 对每个维度单独决定缩放程度
   - 低频维度大幅缩放, 高频维度不缩放
   - 温度修正: mscale = 0.1*ln(s) + 1
   - 效果大幅提升, 几乎不需要微调

4. 实际应用
   - 在 MiniMind 中, 通过 precompute_yarn_freqs_cis 替换
   - 支持从 2K 训练, 扩展到 8K/16K 推理

下一步: 第17课 - mHC (Manifold-Constrained Hyper-Connections)
""")


# ============================================================
# 深入理解：YaRN 的"分频段调速"
# ============================================================
print("\n" + "=" * 60)
print("深入理解：YaRN 的'分频段调速'")
print("=" * 60)

print("""
【类比：YaRN = 望远镜的变焦镜头】
─────────────────────────
  
  训练好的望远镜 (LLaMA, 2K 上下文):
    设计用于看近距离 (短上下文)
    调焦距已固定
  
  想看远 (长上下文):
    简单拉伸: 模糊不清
    换镜头: 麻烦且贵
  
  YaRN:
    智能变焦: 对不同焦段用不同调整
    - 低频 (近): 几乎不调
    - 高频 (远): 大幅调整
    → 效果接近换镜头, 成本极低


【为什么 Transformer 不能直接外推】
──────────────────────────────────

  RoPE 原理回顾:
    频率 θ_i = base^{-2i/d}
    i 越小, θ 越大 (高频, 周期短)
    i 越大, θ 越小 (低频, 周期长)
  
  问题:
    训练时 θ_i 在 [θ_min, θ_max] 范围
    推理时位置 m 超出训练最大位置
    → m * θ_i 变得很大
    → 旋转角度'溢出', 失去区分度
    
  类比:
    钟表的指针每秒转 6°
    看 60 秒 (1 圈) 没问题
    看 3600 秒 (60 圈) 就分不清位置
    → 因为周期太短, 转太多圈


【三种长度外推方法对比】
─────────────────────

  ┌──────────┬────────────┬────────────┬──────────────┐
  │ 方法      │ 核心思想    │ 需要微调    │ 性能          │
  ├──────────┼────────────┼────────────┼──────────────┤
  │ PI       │ 线性调整位置│ 是 (长文本) │ 中            │
  │ NTK-aware│ 调整基础频率│ 否         │ 中            │
  │ YaRN     │ 分频段调整  │ 否/极少    │ 高            │
  └──────────┴────────────┴────────────┴──────────────┘

  PI (Position Interpolation):
    把位置 m 缩放: m' = m * (L_train / L_target)
    → 让推理位置落入训练范围
    → 简单但损失高频信息
    
  NTK-aware:
    调整 base 频率, 让低频变慢
    → 不缩放位置, 改频率
    → 高频不变, 低频更密
    
  YaRN (本课):
    结合 PI + NTK + 注意力缩放
    → 最佳效果


【YaRN 的核心公式】
────────────────

  对每个频率维度 i, 引入缩放函数 r(i):

  h_θ(m, i) = m * r(i) * θ_i

  其中 r(i) 是关键:
    ┌────────────────────────────────────────┐
    │ r(i) =                                 │
    │   1                    if λ(i) < α    │ ← 低频, 不变
    │   (λ(i) - α) / (β - α) if α ≤ λ(i) ≤ β│ ← 中频, 线性插值
    │   1 / s                if λ(i) > β    │ ← 高频, 大幅缩放
    └────────────────────────────────────────┘

  其中:
    λ(i) = 波长 = 2π / θ_i
    s = L_target / L_train (缩放比例)
    α, β: 阈值 (经验值 α=1, β=32)

  类比:
    低频维度 (i 大): 旋转慢, 周期长
      → 拉长位置不会'溢出'
      → 不用调整
    高频维度 (i 小): 旋转快, 周期短
      → 容易'溢出'
      → 必须大幅缩放


【YaRN 三件套】
────────────

  1. NTK-by-parts (频率缩放)
    不同维度用不同的缩放
    → 比统一缩放更精细
  
  2. Attention Scaling (注意力缩放)
    长上下文时, 注意力分数的'温度'要调整
    
    公式: attn / sqrt(t)
    t = 0.1 * ln(s) + 1     (YaRN 经验)
    
    s 越大 (拉得越长), t 越大
    → softmax 越'平滑'
    → 避免过度关注某位置
  
  3. 少量微调 (可选)
    用 0.1% 的长文本微调
    → 让模型适应新分布
    → 大幅提升效果


【YaRN 推理代码 (简化)】
────────────────────

  def precompute_yarn_freqs_cis(
      dim, max_seq_len, base=10000,
      scale_factor=8.0,    # 扩展 8 倍
      alpha=1, beta=32
  ):
      freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
      
      # 1. 计算每个维度的波长
      wavelengths = 2 * pi / freqs
      
      # 2. 分段缩放
      ramp = (wavelengths - alpha) / (beta - alpha)
      ramp = ramp.clamp(0, 1)
      r_inter = (1 - ramp) + ramp / scale_factor
      r_intra = 1 / scale_factor
      
      # 3. 应用缩放
      freqs = freqs * r_inter
      # 高频用 r_intra, 低频用 r_inter
      
      # 4. 注意力缩放
      attn_factor = 0.1 * math.log(scale_factor) + 1.0
      
      return freqs, attn_factor


【YaRN vs 其他方法性能】
──────────────────────

  实验 (LLaMA-7B → 16K 上下文):
  
  ┌──────────────┬──────────┬──────────┐
  │ 方法          │ 困惑度    │ 检索准确率│
  ├──────────────┼──────────┼──────────┤
  │ 无外推        │ > 100   │ 随机      │
  │ PI            │ 6.5     │ 65%       │
  │ NTK-aware     │ 5.8     │ 75%       │
  │ YaRN          │ 4.2     │ 92%       │
  │ YaRN + 微调   │ 3.9     │ 95%       │
  └──────────────┴──────────┴──────────┘

  → YaRN 几乎达到训练极限
  → 检索任务尤其受益


【实际应用选择】
─────────────

  1. 模型训练用 4K, 推理想用 16K:
    → YaRN (scale_factor=4)
  
  2. 模型训练用 2K, 推理想用 32K:
    → YaRN (scale_factor=16)
    → 配合少量微调
  
  3. 模型训练用 8K, 推理想用 128K:
    → YaRN + ReRoPE
    → 可能需要再训练
  
  4. 不确定:
    → 先试 YaRN, 不行再加微调
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】长度外推问题
  模型训练 4K 上下文, 推理 16K 会发生什么? 为什么?

【练习2】PI 缩放因子
  训练 4K 想外推到 16K, s 应该是多少?

【练习3】高频 vs 低频
  RoPE 中, 高频维度和低频维度哪个更需要缩放? 为什么?

【练习4】YaRN 的 α, β
  α, β 控制什么? 调大/调小会有什么影响?

【练习5】注意力缩放
  长上下文为什么要缩放 attention 分数? 不缩放会怎样?

【练习6】YaRN 的局限
  YaRN 能把 4K 模型用到 1M 上下文吗? 有什么限制?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  现象:")
print("    - 模型输出乱码 / 重复")
print("    - perplexity 爆炸 (从 5 涨到 100+)")
print("    - 注意力分数异常 (过度分散或集中)")
print()
print("  原因:")
print("    RoPE 编码: q_i * exp(i * m * θ)")
print("    训练时 m ∈ [0, 4096]")
print("    推理时 m ∈ [0, 16384]")
print()
print("    旋转角度 = m * θ_i")
print("    m 增 4 倍, 角度增 4 倍")
print("    高频维度 (θ 大): 角度溢出, 失去区分")
print("    低频维度 (θ 小): 角度仍合理")
print()
print("  类比:")
print("    钟表秒针 (高频) 转太多圈分不清")
print("    时针 (低频) 转几圈还能区分")

# 练习2
print("\n【练习2 答案】")
print("  s = L_target / L_train = 16384 / 4096 = 4")
print()
print("  PI 方法:")
print("    位置缩放: m' = m / s = m / 4")
print("    → 把 [0, 16384] 映射到 [0, 4096]")
print("    → 落入训练范围")
print()
print("  YaRN 方法:")
print("    s=4 用作高频维度的缩放因子")
print("    低频维度缩放更少 (或不缩放)")
print()
print("  实际:")
print("    4 倍外推相对容易, YaRN 可行")
print("    16 倍外推比较极限, 需配合微调")
print("    64 倍以上基本不可能")

# 练习3
print("\n【练习3 答案】")
print("  高频维度更需要缩放")
print()
print("  频率分析:")
print("    高频 (i 小, θ 大): 周期短")
print("    低频 (i 大, θ 小): 周期长")
print()
print("  示例 (LLaMA, dim=128, base=10000):")
print("    i=0:   θ = 1.0,       周期 = 2π/1.0 ≈ 6.28")
print("    i=32:  θ ≈ 0.01,      周期 = 2π/0.01 ≈ 628")
print("    i=63:  θ ≈ 0.000103,  周期 ≈ 61000")
print()
print("  问题:")
print("    高频 i=0, 周期 ≈ 6")
print("    位置 m=4096, 旋转 4096/6 ≈ 683 圈")
print("    位置 m=16384, 旋转 16384/6 ≈ 2730 圈")
print("    → 多转 4 倍圈数, 角度无法区分")
print()
print("    低频 i=32, 周期 ≈ 628")
print("    位置 m=4096, 旋转 4096/628 ≈ 6.5 圈")
print("    位置 m=16384, 旋转 16384/628 ≈ 26 圈")
print("    → 仍在合理范围, 角度可区分")
print()
print("  YaRN 处理:")
print("    高频 (i 小): 大幅缩放 (1/s)")
print("    低频 (i 大): 不缩放 (1)")

# 练习4
print("\n【练习4 答案】")
print("  α, β 是 YaRN 缩放函数的分段阈值")
print()
print("  α (下限):")
print("    波长小于 α 的维度: 不缩放 (r=1)")
print("    默认 α=1, 短波 (高频)")
print("    调大 α: 更多维度被识别为'高频', 更多缩放")
print()
print("  β (上限):")
print("    波长大于 β 的维度: 完全缩放 (r=1/s)")
print("    默认 β=32, 长波 (低频)")
print("    调小 β: 更多维度被识别为'低频', 更少缩放")
print()
print("  区间 [α, β]:")
print("    线性插值, 平滑过渡")
print("    避免硬切换带来的不稳定")
print()
print("  实验调优:")
print("    s=4:  α=1, β=32 效果不错")
print("    s=8:  α=1, β=64 可能更好")
print("    s=16: α=1, β=128")
print()
print("  一般原则:")
print("    不需要严格调, 默认值已能 work")
print("    微调对极长上下文有帮助")

# 练习5
print("\n【练习5 答案】")
print("  长上下文注意力分数需要缩放的原因:")
print()
print("  现象:")
print("    训练时: 上下文 4K, attention 分布合理")
print("    推理时: 上下文 16K, attention 过度分散")
print("    → 模型'注意力涣散', 抓不住重点")
print()
print("  数学:")
print("    softmax 分数 = exp(q·k / sqrt(d))")
print("    → 上下文越长, 相关位置越多")
print("    → softmax 后概率被分摊")
print("    → 最高概率也变低")
print()
print("  解决:")
print("    缩放因子 t = 0.1 * ln(s) + 1")
print("    s=4 时 t ≈ 1.14")
print("    s=16 时 t ≈ 1.33")
print("    attn = attn / t")
print()
print("  效果:")
print("    分数更'尖锐'")
print("    → 模型更关注相关位置")
print("    → 检索能力恢复")
print()
print("  不缩放的后果:")
print("    - 检索准确率下降")
print("    - 长文 QA 失效")
print("    - perplexity 升高")

# 练习6
print("\n【练习6 答案】")
print("  YaRN 把 4K 推到 1M: 基本不可能")
print()
print("  限制因素:")
print()
print("  1. 缩放比例极限:")
print("    s = 1M / 4K = 256")
print("    → 实际能用 s ≤ 16 (4 倍外推)")
print("    → s=256 完全不行")
print()
print("  2. 注意力机制本身:")
print("    注意力需要 O(N) 显存")
print("    1M 上下文, 单 batch 需要 1M² = 1T 元素")
print("    → 显存根本装不下")
print()
print("  3. 模型训练时没见过长距离:")
print("    RoPE 是相对位置, 但绝对位置也在隐式学习")
print("    极端长距离时, 模型'认知'超限")
print()
print("  实用方案:")
print("    4K → 16K:   YaRN 容易, 几乎无损")
print("    4K → 32K:   YaRN + 少量微调")
print("    4K → 64K:   YaRN + 显著微调")
print("    4K → 128K+: 需要配合其他技术")
print()
print("  其他技术:")
print("    - Ring Attention: 多卡分片")
print("    - 滑动窗口: Mistral 风格")
print("    - 记忆压缩: RAG / MemGPT")
print("    - 状态空间: Mamba")
print()
print("  工业界典型:")
print("    - Mistral 7B: 训练 8K, 实际可用 32K+")
print("    - LLaMA-2: 训练 4K, 通过技术推到 200K+")
print("    - Kimi: 训练 128K, 推到更长")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
