"""
第6课：FFN — 前馈网络：注意力之后的记忆提炼
===============================================

Attention 让词与词交换信息，但每个位置本身的表达能力有限。
FFN (Feed-Forward Network) 的作用：对每个位置独立地进行"记忆提炼"。

核心结构（SwiGLU 变体，MiniMind 使用）：
  FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

拆解：
  1. gate_proj: 把 hidden_size 维向量映射到更高维（intermediate_size）
  2. SiLU 激活: gate_proj 的输出过激活函数（门控信号）
  3. up_proj: 另一个投影到高维（信息路径）
  4. 逐元素相乘: 门控 * 信息 = 选择性保留
  5. down_proj: 映射回 hidden_size 维

直觉理解：
  Attention = 词与词之间的"交流"
  FFN = 每个词自己的"思考"

  Attention 收集了上下文信息后，FFN 对每个位置独立加工，
  把重要信息提炼存储，把噪声过滤掉。

运行: python lessons/lesson06_ffn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一步：最简单的 FFN
# ============================================================

print("=" * 60)
print("实验1：最简单的 FFN — 两层线性变换")
print("=" * 60)

class SimpleFFN(nn.Module):
    """最简单的 FFN：Linear → ReLU → Linear"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x)
        x = self.down_proj(x)
        return x

hidden_size = 64
intermediate_size = 256

ffn = SimpleFFN(hidden_size, intermediate_size)
x = torch.randn(1, 8, hidden_size)
out = ffn(x)

print(f"输入形状: {x.shape}")
print(f"升维后:   [{x.shape[0]}, {x.shape[1]}, {intermediate_size}]")
print(f"输出形状: {out.shape}")
print(f"\nFFN 的维度变化: {hidden_size} → {intermediate_size} → {hidden_size}")
print(f"先升维（展开信息），再降维（压缩提炼）")


# ============================================================
# 第二步：激活函数的作用
# ============================================================

print("\n" + "=" * 60)
print("实验2：为什么需要激活函数？")
print("=" * 60)

x = torch.linspace(-3, 3, 100)

# ReLU: max(0, x)
relu_out = F.relu(x)

# GELU: x * Φ(x)，比 ReLU 更平滑
gelu_out = F.gelu(x)

# SiLU (Swish): x * sigmoid(x)，MiniMind 使用
silu_out = F.silu(x)

print("激活函数对比:")
print(f"  ReLU: max(0, x) — 简单粗暴，负数直接归零")
print(f"  GELU: x * Φ(x) — 平滑版 ReLU，负数不完全归零")
print(f"  SiLU: x * σ(x) — 自门控，x 乘以自己的概率")
print(f"\n  x=-2: ReLU={F.relu(torch.tensor(-2.0)):.4f}, "
      f"GELU={F.gelu(torch.tensor(-2.0)):.4f}, "
      f"SiLU={F.silu(torch.tensor(-2.0)):.4f}")
print(f"  x=0:  ReLU={F.relu(torch.tensor(0.0)):.4f}, "
      f"GELU={F.gelu(torch.tensor(0.0)):.4f}, "
      f"SiLU={F.silu(torch.tensor(0.0)):.4f}")
print(f"  x=2:  ReLU={F.relu(torch.tensor(2.0)):.4f}, "
      f"GELU={F.gelu(torch.tensor(2.0)):.4f}, "
      f"SiLU={F.silu(torch.tensor(2.0)):.4f}")

# 没有激活函数会怎样？
linear_no_act = nn.Sequential(
    nn.Linear(hidden_size, intermediate_size, bias=False),
    nn.Linear(intermediate_size, hidden_size, bias=False),
)
linear_with_act = SimpleFFN(hidden_size, intermediate_size)

x_test = torch.randn(1, 4, hidden_size)
out_no_act = linear_no_act(x_test)
out_with_act = linear_with_act(x_test)

print(f"\n无激活函数: 两个线性层 = 一个线性层（等价于 hidden_size → hidden_size）")
print(f"有激活函数: 引入非线性，表达能力远超单一线性层")


# ============================================================
# 第三步：SwiGLU — MiniMind 使用的 FFN 变体
# ============================================================

print("\n" + "=" * 60)
print("实验3：SwiGLU — 门控线性单元")
print("=" * 60)

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN（MiniMind 使用的方式）

    FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    与简单 FFN 的区别：
      - 多了一个 gate_proj（门控路径）
      - SiLU(gate) * up 实现选择性信息传递
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))   # 门控信号：决定保留多少信息
        up = self.up_proj(x)                # 信息路径：要传递的内容
        return self.down_proj(gate * up)    # 门控 * 信息 → 选择性保留


ffn_simple = SimpleFFN(hidden_size, intermediate_size)
ffn_swiglu = SwiGLUFFN(hidden_size, intermediate_size)

x = torch.randn(1, 8, hidden_size)
out_simple = ffn_simple(x)
out_swiglu = ffn_swiglu(x)

simple_params = sum(p.numel() for p in ffn_simple.parameters())
swiglu_params = sum(p.numel() for p in ffn_swiglu.parameters())

print(f"简单 FFN 参数量: {simple_params:,}")
print(f"SwiGLU FFN 参数量: {swiglu_params:,}")
print(f"SwiGLU 多了 gate_proj，参数量约 {swiglu_params/simple_params:.1f}x")

# 可视化门控机制
torch.manual_seed(42)
x_demo = torch.randn(1, 1, hidden_size)
gate = F.silu(ffn_swiglu.gate_proj(x_demo))
up = ffn_swiglu.up_proj(x_demo)

print(f"\n门控信号 (gate_proj → SiLU):")
print(f"  范围: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
print(f"  均值: {gate.mean().item():.4f}")
print(f"  SiLU 输出接近0 -> 该维度被'关闭'")
print(f"  SiLU 输出较大 -> 该维度被'打开'")

gated = gate * up
print(f"\n门控后 (gate * up):")
print(f"  范围: [{gated.min().item():.4f}, {gated.max().item():.4f}]")
print(f"  门控实现了选择性信息传递！")


# ============================================================
# 第四步：FFN 的参数量分析
# ============================================================

print("\n" + "=" * 60)
print("实验4：FFN 是模型参数的大头")
print("=" * 60)

configs = [
    ("MiniMind-Small", 768, 2048),
    ("MiniMind-Medium", 1024, 2816),
    ("LLaMA-7B", 4096, 11008),
]

for name, hidden, intermediate in configs:
    gate_params = hidden * intermediate
    up_params = hidden * intermediate
    down_params = intermediate * hidden
    total_ffn = gate_params + up_params + down_params
    total_attn = 4 * hidden * hidden  # Q, K, V, O projections (simplified)

    print(f"\n{name}: hidden={hidden}, intermediate={intermediate}")
    print(f"  gate_proj: {hidden}×{intermediate} = {gate_params:,}")
    print(f"  up_proj:   {hidden}×{intermediate} = {up_params:,}")
    print(f"  down_proj: {intermediate}×{hidden} = {down_params:,}")
    print(f"  FFN 总参数: {total_ffn:,}")
    print(f"  Attention 总参数 (简化): {total_attn:,}")
    print(f"  FFN/Attention 比例: {total_ffn/total_attn:.1f}x")

print("\n→ FFN 的参数量通常是 Attention 的 2-3 倍！")
print("→ intermediate_size 是 hidden_size 的约 2.7-3 倍")


# ============================================================
# 第五步：FFN 的逐位置独立性
# ============================================================

print("\n" + "=" * 60)
print("实验5：FFN 对每个位置独立操作")
print("=" * 60)

ffn = SwiGLUFFN(64, 256)
x = torch.randn(1, 4, 64)

# 方式1：整体输入
out_batch = ffn(x)

# 方式2：逐位置输入
out_positions = []
for i in range(4):
    pos_out = ffn(x[:, i:i+1, :])
    out_positions.append(pos_out)
out_manual = torch.cat(out_positions, dim=1)

print(f"整体输入输出: {out_batch.shape}")
print(f"逐位置输入输出: {out_manual.shape}")
print(f"两种方式结果一致: {torch.allclose(out_batch, out_manual, atol=1e-5)}")
print("\n→ FFN 对每个位置独立操作，不涉及位置间的信息交换")
print("→ 这与 Attention 形成互补：Attention 交流，FFN 思考")


# ============================================================
# 第六步：FFN 作为"键值记忆"
# ============================================================

print("\n" + "=" * 60)
print("实验6：FFN 的记忆视角")
print("=" * 60)

print("""
研究发现，FFN 可以看作一个"键值记忆系统"：

  up_proj 的每一行 = 一个"键"（key pattern）
  down_proj 的每一列 = 一个"值"（value pattern）

  FFN(x) = Σ  activation(key_i · x) * value_i

  即：输入 x 与所有 key 计算匹配度（激活值），
  然后用匹配度加权求和对应的 value。

【逐步拆解：FFN 为什么是 Key-Value Memory？】

  第1步：看 FFN 的数学形式
    FFN(x) = W_down · σ(W_up · x)

  第2步：把矩阵拆成行向量
    W_up 的第 i 行 = key_i（一个"模式检测器"）
    W_down 的第 i 列 = value_i（一个"知识存储"）

  第3步：展开计算
    h = W_up · x  →  h_i = key_i · x  （输入与第i个key的匹配度）
    a = σ(h)      →  a_i = σ(key_i · x)（匹配度过激活函数，决定"激活多少"）
    y = W_down · a →  y = Σ a_i · value_i（按激活程度加权求和value）

  第4步：直觉理解
    key_i = "这是主语位置"的模式 → value_i = "应该填入名词的特征"
    key_j = "这是否定位置"的模式 → value_j = "应该反转情感的特征"

    输入 x 如果匹配 key_i，就激活 value_i
    输入 x 如果匹配 key_j，就激活 value_j
    最终输出 = 所有被激活的 value 的加权和

  第5步：与 Attention 的对比
    Attention：Q·K^T 找"哪个词和我相关" → 加权 V
    FFN：      x·key_i 找"哪个知识和我相关" → 加权 value_i

    Attention 是"词与词"之间的记忆查找（动态的，每次输入不同）
    FFN 是"输入与知识"之间的记忆查找（静态的，权重是学到的知识）

  类比：
    Attention = 你在会议上听别人发言，选择关注谁
    FFN = 你在图书馆查资料，选择读哪些书
""")


# ============================================================
# 第七步：MiniMind 中的 FFN 配置
# ============================================================

print("=" * 60)
print("实验7：MiniMind 中的 FFN 配置")
print("=" * 60)

# MiniMind 实际配置
print("MiniMind 的 FFN 配置:")
print("  hidden_size = 768")
print("  intermediate_size = 2048 (约 2.67x hidden_size)")
print("  激活函数 = SiLU (SwiGLU 变体)")
print("  无偏置 (bias=False)")
print()
print("每个 Transformer Block 的 FFN 参数量:")
hidden = 768
inter = 2048
params = 3 * hidden * inter  # gate + up + down
print(f"  3 × {hidden} × {inter} = {params:,} = {params/1e6:.2f}M")
print(f"  (gate_proj + up_proj + down_proj)")


# ============================================================
# 深入理解：FFN 的本质
# ============================================================
print("\n" + "=" * 60)
print("深入理解：FFN 的本质与类比")
print("=" * 60)

print("""
【类比：FFN 是"知识库检索"】
─────────────────────────
可以把 FFN 看作一个巨大的"键值记忆":

  down_proj 权重 W_down: [hidden × intermediate]
  → 每一行 = 一个"知识条目"
  → 中间维度 = 知识条目的数量

  流程:
    1. gate_proj 决定"我关心什么知识" (查询)
    2. up_proj 把输入映射到"知识空间"
    3. SiLU(gate) * up = "按相关性过滤知识"
    4. down_proj 把"激活的知识"还原成 hidden 表示

  类比图书馆:
    假设 down_proj 的每一行是一本书的内容
    输入 x 决定"我想读哪些书"
    gate 决定"这些书的相关性"
    输出 = 混合相关书的内容


【图示：SwiGLU FFN 的数据流】
─────────────────────────────

       x (hidden_size)
       │
       ├──────────────┐
       │              │
       ▼              ▼
   ┌────────┐    ┌────────┐
   │gate_proj│   │up_proj  │
   └────┬────┘   └────┬────┘
        │             │
        ▼             ▼
       SiLU           ×
        │             │
        └─────⊙───────┘  ← 逐元素相乘 (门控)
              │
              ▼
         ┌────────┐
         │down_proj│
         └────┬────┘
              │
              ▼
        output (hidden_size)


【为什么升维？】
──────────────
  类比: 解决问题时, 先发散思维 (考虑多种可能)
       再聚焦 (选出最相关的)

  数学上:
    - 低维空间: 表达力有限, 难以分离不同概念
    - 高维空间: 表达力强, 可以学习复杂映射
    - 中间维度通常是 hidden 的 2.67-4 倍
    - MiniMind 用 4倍 (例如 hidden=512, inter=2048)

  经验值: 大模型倾向于更大的中间维度
    GPT-3: 4x hidden
    LLaMA: (8/3)x hidden (swiGLU)
    PaLM:  4x hidden


【激活函数对比】
──────────────
  ReLU:     f(x) = max(0, x)
            简单, 但负值直接归零 (神经元死亡)
  
  GELU:     f(x) = x * Φ(x) (高斯)
            平滑, 训练更稳定, BERT 用
  
  SiLU:     f(x) = x * sigmoid(x) (Swish)
            LLaMA / MiniMind 用
            比 ReLU 平滑, 比 GELU 略快
            在大模型上效果略好于 GELU

  SwiGLU:   f(x, gate) = SiLU(gate) * x
            在 SiLU 基础上加门控
            当前大模型的事实标准
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】激活函数必要性
  如果 FFN 中没有激活函数 (只做线性变换), 会怎样?

【练习2】FFN 参数量
  hidden=512, inter=2048, SwiGLU 的参数量是多少?
  相当于 hidden 大小的多少倍?

【练习3】FFN vs Attention 参数量
  设 hidden=512, num_heads=8, head_dim=64
  单层 Attention 参数量 = ?
  单层 FFN 参数量 = ?
  FFN 是 Attention 的几倍?

【练习4】门控机制
  SwiGLU 相比普通 FFN 多了一个 gate_proj
  多出的参数有什么用?

【练习5】FFN 思考 vs Attention 交流
  模型中 Attention 和 FFN 的分工是什么?
  如果删掉 FFN, 模型会怎样?

【练习6】为什么 down_proj 要乘以 2/3?
  LLaMA 论文说 SwiGLU 的中间维度应该是 (2/3) * 4 * hidden
  为什么要乘以 2/3?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  没有激活函数, 多层线性 = 一层线性")
print("  数学证明:")
print("    y = W2(W1·x + b1) + b2 = W2·W1·x + (W2·b1 + b2)")
print("    令 W = W2·W1, b = W2·b1 + b2")
print("    y = W·x + b, 等价于单层线性")
print()
print("  深层模型会退化为浅层模型, 无法学到复杂关系")
print("  → 激活函数是深度学习的'灵魂'")

# 演示
import torch
import torch.nn as nn
linear = nn.Sequential(
    nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8),
    nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 4)
)
total = sum(p.numel() for p in linear.parameters())
print(f"  6 层纯线性网络参数量: {total}")

# 等价
single = nn.Linear(8, 4)
print(f"  等价单层参数量: {sum(p.numel() for p in single.parameters())}")
print(f"  6 层没用, 等价 1 层!")

# 练习2
print("\n【练习2 答案】")
hidden = 512
inter = 2048
gate = hidden * inter
up = hidden * inter
down = inter * hidden
total = gate + up + down
print(f"  gate_proj: {hidden} × {inter} = {gate:,}")
print(f"  up_proj:   {hidden} × {inter} = {up:,}")
print(f"  down_proj: {inter} × {hidden} = {down:,}")
print(f"  合计: {total:,} = {total/1e6:.2f}M")
print(f"  相当于 {total/(hidden*hidden):.1f}x hidden² = {(total/3)/(hidden*hidden):.1f}x 单层 Linear")

# 练习3
print("\n【练习3 答案】")
hidden = 512
num_heads = 8
head_dim = 64
inter = 2048

# Attention: Q, K, V, O 四个 proj
attn_params = 4 * hidden * hidden
print(f"  Attention: 4 × {hidden}² = {attn_params:,} = {attn_params/1e6:.2f}M")

# FFN: gate, up, down 三个 proj (SwiGLU)
ffn_params = 3 * hidden * inter
print(f"  FFN (SwiGLU): 3 × {hidden} × {inter} = {ffn_params:,} = {ffn_params/1e6:.2f}M")

print(f"  FFN / Attention = {ffn_params/attn_params:.2f}x")
print(f"  → FFN 参数量是 Attention 的 ~3 倍, 是模型参数的大头")

# 练习4
print("\n【练习4 答案】")
print("  多出的 gate_proj 起到'选择性激活'的作用")
print()
print("  普通 FFN:  y = activation(W1·x) * W2")
print("             ↑ 全部维度都被激活")
print()
print("  SwiGLU:    y = (SiLU(W_gate·x) * W_up·x) * W_down")
print("             ↑ SiLU 输出接近 0 的维度, 整个通道被'关闭'")
print("             ↑ 接近 1 的维度, 信息完全通过")
print()
print("  门控的效果:")
print("    - 强制 FFN 学到'该激活哪些维度'")
print("    - 减少无用计算 (被门控过滤掉)")
print("    - 提升模型表达力 (sparse activation)")
print("    - 实验上 SwiGLU > GELU > ReLU")

# 练习5
print("\n【练习5 答案】")
print("  Attention 的作用: 信息交流")
print("    - 让不同位置交换信息")
print("    - 每个位置的输出 = 加权平均所有位置")
print("    - 复杂度: O(N²) (N 是序列长度)")
print()
print("  FFN 的作用: 信息加工")
print("    - 对每个位置独立加工 (不涉及位置间通信)")
print("    - 提取、变换、记忆特征")
print("    - 复杂度: O(N) (每个位置一次)")
print()
print("  如果删掉 FFN:")
print("    - 模型只剩 '信息混合', 没有 '信息加工'")
print("    - 性能大幅下降 (~50% loss)")
print("    - 类似只有图书馆, 没有大脑")
print()
print("  二者缺一不可, 共同构成 Transformer Block")
print("  → Attention 让信息流动, FFN 让信息'思考'")

# 练习6
print("\n【练习6 答案】")
print("  普通 FFN 的中间维度 = 4 * hidden")
print("  SwiGLU 多了一个 gate_proj, 总参数量增加了")
print()
print("  为了保持总参数量不变, 调整中间维度:")
print("    普通:  2 * hidden * inter = 2 * hidden * (4*hidden) = 8 * hidden²")
print("    SwiGLU: 3 * hidden * inter (gate + up + down)")
print("    令 3 * hidden * inter = 8 * hidden²")
print("    inter = 8/3 * hidden")
print()
print("  论文 (Shazeer 2020) 通过实验发现:")
print("    inter = 2/3 * 4 * hidden = 8/3 * hidden 时")
print("    SwiGLU 在相同参数量下效果最好")
print()
print("  → 这是工程上的'参数预算平衡'技巧")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. FFN 对每个位置独立加工：升维 → 激活 → 降维
2. 激活函数引入非线性，没有它两层线性=一层线性
3. SwiGLU = SiLU(gate) * up，门控机制选择性保留信息
4. FFN 参数量是 Attention 的 2-3 倍，是模型的大头
5. FFN 可看作"键值记忆"：key 匹配输入，value 提供知识
6. Attention 交流 + FFN 思考 = Transformer 的两大支柱

数据流至此：
  ... → Attention → 残差连接 → RMSNorm → FFN → 残差连接 → ...
         词间交流     加上原始信息   归一化    位置加工   加上原始信息

下一步 → lesson07_block.py：把 Attention + FFN 组装成 Transformer Block
""")
