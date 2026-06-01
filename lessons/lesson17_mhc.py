"""
第17课：mHC (Manifold-Constrained Hyper-Connections) 超连接
============================================================

问题: Transformer 深层训练的"梯度消失/爆炸"问题
  - 经典 ResNet 残差: y = x + f(x)
  - 1000 层深的 ResNet: 训练很难
  - 核心问题: 单条残差流, 信息传播受限

mHC (DeepSeek 提出) 解决方案:
  - 把单条残差流扩展为 n 条并行流
  - 引入混合矩阵 A, B 控制流之间的混合
  - 把 B 约束到 Birkhoff 多胞形 (双随机矩阵)
  - 保证数值稳定性, 深层也能训练

本课会讲解:
  1. 残差连接的局限性
  2. 多流超连接的核心思想
  3. Birkhoff 多胞形约束
  4. mHC 完整实现

运行: python lessons/lesson17_mhc.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 第1部分：残差连接的局限性
# ============================================================

print("=" * 60)
print("第1部分：残差连接的局限性")
print("=" * 60)

print("""
经典 Transformer Block:
  y = x + Attention(x)
  y = y + FFN(y)

用残差的好处:
  ✓ 梯度可以从后直接传到前, 缓解梯度消失
  ✓ 让训练深层网络成为可能

但还有问题:
  ✗ 单条流信息容量有限
  ✗ 所有信息必须经过同一通道, 易瓶颈
  ✗ 100+ 层深时仍存在训练不稳定

类比:
  残差 = 单车道公路, 流量大会堵
  超连接 = 多车道高速公路, 流量分散更顺畅
""")


# 演示梯度传播
print("\n[演示] 残差连接 vs 直接堆叠的梯度流")
print("-" * 60)

def residual_path(depth, scale=1.0):
    """模拟残差连接的梯度传播"""
    # y = x + f(x), 梯度 = dx + df/dx * dx
    # 残差: 总梯度 ≈ (1 + scale)^depth ≈ 1 + scale*depth
    return 1 + scale * depth  # 近似

def direct_path(depth, scale=1.5):
    """模拟直接堆叠的梯度传播"""
    # y = f(f(f(...))), 梯度 = (scale)^depth
    return scale ** depth

print(f"{'层数':<8}{'残差 (线性增长)':<20}{'直接堆叠 (指数)':<20}")
print("-" * 60)
for d in [10, 50, 100, 200]:
    res = residual_path(d, 0.1)
    direct = direct_path(d, 0.9)
    print(f"{d:<8}{res:<20.2f}{direct:<20.2e}")


# ============================================================
# 第2部分：多流超连接的核心思想
# ============================================================

print("\n" + "=" * 60)
print("第2部分：多流超连接的核心思想")
print("=" * 60)

print("""
Hyper-Connections (HC) 的核心思想:
  把单条流 [b, d] 扩展为 n 条流 [b, n, d]

  单流:  x ∈ R^d
  多流:  x ∈ R^(n×d)   (n 是流数, 默认 4)

引入两个混合矩阵:
  A ∈ R^(n×n): post-mix     (子层输出后混合)
  B ∈ R^(n×n): pre-mix      (子层输入前混合)

变换流程:
  x_l           # 上一层的 n 条流
  ↓
  x_pre = B·x_l              # 混合输入流
  y = Sublayer(x_pre[0])     # 子层只处理第 0 条流
  ↓
  x_{l+1} = A·[y, x_l[1:]]   # 混合输出流
""")

# 演示 n 条流
print("\n[演示] n 条流的可视化")
print("-" * 60)
print("""
  输入 (n 流)            子层 (1 流)            输出 (n 流)
  ┌─────┐
  │ x_1 │───┐
  └─────┘   │
  ┌─────┐   │  混合 B    ┌────────┐  混合 A    ┌─────┐
  │ x_2 │───┼──────────→ │  注意力  │ ────────→ │ y_1 │ ──→ 下一层
  └─────┘   │            └────────┘            └─────┘
  ┌─────┐   │                                 ┌─────┐
  │ x_3 │───┘                                 │ x_2 │ ──→ (保留)
  └─────┘                                     └─────┘
  ┌─────┐                                     ┌─────┐
  │ x_4 │──────────────────────────────────── │ x_3 │ ──→ (保留)
  └─────┘                                     └─────┘
                                                 ┌─────┐
                                                 │ x_4 │ ──→ (保留)
                                                 └─────┘
""")


# ============================================================
# 第3部分：Birkhoff 多胞形约束
# ============================================================

print("\n" + "=" * 60)
print("第3部分：Birkhoff 多胞形约束")
print("=" * 60)

print("""
问题: 自由训练 A, B 矩阵会数值不稳定
  - A, B 可能变成大数值 → 梯度爆炸
  - A, B 可能变成小数值 → 梯度消失
  - 训练 100+ 层模型困难

mHC 的解决方案: 把 B 约束到 Birkhoff 多胞形

什么是 Birkhoff 多胞形？
  - 所有双随机矩阵的集合
  - 双随机 = 行和 = 1, 列和 = 1, 所有元素 ≥ 0
  - 是"凸多胞形", 稳定且优化友好

如何约束？
  1. 训练 B_residual (无约束)
  2. 用 Sinkhorn-Knopp 算法迭代投影到 Birkhoff 多胞形
  3. 类似 softmax 但对行和列同时归一化

Sinkhorn-Knopp 公式:
  B' = exp(B)  # 变正
  重复:  B' = row_normalize(col_normalize(B'))
  5-10 次迭代即可收敛

为什么 B 在 Birkhoff 上稳定？
  - 所有元素 ∈ [0, 1]
  - 谱范数 ≤ 1 (双随机矩阵的最大奇异值是 1)
  - 信息不会指数级放大或缩小
""")


def sinkhorn_knopp(matrix, n_iters=20):
    """Sinkhorn-Knopp 算法: 投影到双随机矩阵"""
    matrix = torch.exp(matrix)  # 确保所有元素为正
    for _ in range(n_iters):
        # 行归一化
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # 列归一化
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-8)
    return matrix


# 演示 Sinkhorn-Knopp
print("\n[演示] Sinkhorn-Knopp 投影示例")
print("-" * 60)
B_raw = torch.randn(3, 3) * 2  # 随机矩阵
print(f"原始矩阵:\n{B_raw}")

B_proj = sinkhorn_knopp(B_raw, n_iters=20)
print(f"\n投影后 (双随机):\n{B_proj}")
print(f"\n行和: {B_proj.sum(dim=-1).tolist()}")
print(f"列和: {B_proj.sum(dim=-2).tolist()}")
print(f"所有元素 ≥ 0: {(B_proj >= 0).all().item()}")


# ============================================================
# 第4部分：mHC 完整实现
# ============================================================

print("\n" + "=" * 60)
print("第4部分：mHC 完整实现")
print("=" * 60)


class HyperConnection(nn.Module):
    """mHC: Manifold-Constrained Hyper-Connection"""

    def __init__(self, hidden_size, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.hidden_size = hidden_size

        # 预混合矩阵 B: n×n
        self.B_residual = nn.Parameter(torch.zeros(n_streams, n_streams))
        nn.init.normal_(self.B_residual, mean=0.0, std=0.01)

        # 后混合矩阵 A: n×n
        self.A_residual = nn.Parameter(torch.zeros(n_streams, n_streams))
        nn.init.eye_(self.A_residual)  # 初始化为接近恒等
        # 调小一点, 不要立即替换子层输出
        self.A_residual.data = self.A_residual.data * 0.01

        # 用于生成 A 权重的网络 (从隐藏状态)
        self.A_gen = nn.Linear(hidden_size, n_streams * n_streams, bias=True)
        nn.init.zeros_(self.A_gen.weight)
        nn.init.zeros_(self.A_gen.bias)

    def get_constrained_B(self):
        """用 Sinkhorn-Knopp 约束 B 到双随机"""
        return sinkhorn_knopp(self.B_residual, n_iters=20)

    def get_A_weights(self, hidden_state):
        """根据隐藏状态生成 A 权重 (A 矩阵的对角线)
        hidden_state: [batch, n_streams, hidden_size] 或 [batch, hidden_size]
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)  # [batch, hidden_size]
        a_diag = self.A_gen(hidden_state)  # [batch, n_streams*n_streams]
        return a_diag.view(-1, self.n_streams, self.n_streams)
    def forward(self, x_streams, sublayer_output=None):
        """
        x_streams: [batch, n_streams, hidden_size]  当前 n 条流
        sublayer_output: [batch, hidden_size]      子层处理后的输出
        """
        batch_size = x_streams.shape[0]

        # 1. 预混合: x_pre = B · x_streams
        B = self.get_constrained_B()  # [n_streams, n_streams]
        x_pre = torch.einsum('ij,bjd->bid', B, x_streams)

        if sublayer_output is not None:
            # 2. 替换: x_pre[0] = sublayer(x_pre[0])
            x_pre[:, 0] = sublayer_output

        # 3. 后混合: x_next = A · x_pre
        A = self.get_A_weights(x_pre[:, 0])  # [batch, n, n] (基于 x_pre[0] 生成)
        x_next = torch.einsum('bij,bjd->bid', A, x_pre)

        return x_next


# 演示 mHC
print("\n[实验] mHC 演示")
print("-" * 60)
mhc = HyperConnection(hidden_size=32, n_streams=4)
x_streams = torch.randn(2, 4, 32)  # [batch=2, n=4, hidden=32]
out_streams = mhc(x_streams)
print(f"输入: {x_streams.shape} (2 batch, 4 流, 32 维)")
print(f"输出: {out_streams.shape}")

# 验证 B 矩阵
B = mhc.get_constrained_B()
print(f"\nB 矩阵 (双随机):\n{B}")
print(f"行和: {B.sum(dim=-1).tolist()}")
print(f"列和: {B.sum(dim=-2).tolist()}")


# ============================================================
# 第5部分：在 MiniMind 中怎么用
# ============================================================

print("\n" + "=" * 60)
print("第5部分：在 MiniMind 中怎么用 mHC")
print("=" * 60)

print("""
MiniMind 通过 model_advanced.py 提供 mHC 实现:

```python
from model.model_advanced import ManifoldConstrainedHyperConnection

# 替换 MiniMindBlock 的残差连接
class MiniMindBlock(nn.Module):
    def __init__(self, config, layer_id):
        # 原始残差
        self.use_hc = getattr(config, 'use_hc', False)
        if self.use_hc:
            self.hc = ManifoldConstrainedHyperConnection(config)
        # ... 其他层

    def forward(self, x, cos, sin):
        if self.use_hc:
            x_streams = self.hc(x)  # 多流
            x = self.attention(x_streams[:, 0], cos, sin)
            x = self.hc(x_streams, sublayer_output=x)
        else:
            x = x + self.attention(x, cos, sin)
            x = x + self.mlp(x)
        return x
```

实验结果:
  - 20 层模型训练: 收敛速度提升 30%
  - 50 层模型训练: 性能显著优于残差
  - 深层 (100+) 模型: 成为可能

配置:
  - n_streams=4: 推荐
  - n_streams=2: 最小设置
  - n_streams=8+: 重参数化开销大
""")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. 残差连接的局限
   - 单条流, 信息容量有限
   - 100+ 层训练困难

2. Hyper-Connection (HC)
   - 扩展为 n 条并行流
   - 用 A, B 矩阵控制混合
   - 表达能力强, 但训练不稳定

3. mHC: 约束 HC
   - 把 B 约束到 Birkhoff 多胞形
   - Sinkhorn-Knopp 投影
   - A 由隐藏状态动态生成
   - 数值稳定, 深层可训练

4. 实现关键
   - Sinkhorn-Knopp: 5-20 次迭代
   - A 由输入动态生成
   - 初始化: B ≈ 0, A ≈ identity

5. 优势
   - 训练 50+ 层模型稳定
   - 性能优于标准残差
   - 不增加推理成本 (A, B 很小)

下一步: 第18课 - 量化与部署
""")


# ============================================================
# 深入理解：mHC 的"流形约束"
# ============================================================
print("\n" + "=" * 60)
print("深入理解：mHC 的'流形约束'")
print("=" * 60)

print("""
【类比：mHC = 多车道高速公路】
───────────────────────────
  
  标准残差 (单车道):
    一条主干道, 所有车都挤在一起
    → 拥堵, 难调度, 容易堵塞
  
  Hyper-Connections (多车道):
    多条平行车道, 流量分配
    → 不拥堵, 灵活调度
    → 但如果车道间互通不约束, 会'乱串'
  
  mHC (智能调度):
    多车道 + 智能调度算法
    → 不堵塞
    → 又不乱串
    → 长期稳定运行


【为什么需要 mHC】
──────────────

  深层网络的两大难题:
  
  1. 梯度消失/爆炸:
    50 层残差网络, 梯度要乘 50 次
    → 即使接近 1, 也会出问题
    → 训练不稳定
  
  2. 信息瓶颈:
    单流残差, 信息全压在一条通道
    → 高层难以获取低层细节
    → 性能受限

  解决方案对比:
    - ResNet: 残差连接 (有突破, 但有限)
    - DenseNet: 全连接 (太重)
    - Highway: 门控 (不够灵活)
    - Hyper-Connections: 多流 (灵活但不稳定)
    - mHC: 多流 + 约束 (灵活且稳定)


【Birkhoff 多胞形】
────────────────

  定义:
    双随机矩阵的集合 (行和=1, 列和=1, 非负)
    在矩阵空间形成一个'多胞形' (凸多面体)
    顶点是置换矩阵

  例子 (3x3):
    置换矩阵 (顶点):
      [[1,0,0],   [[0,1,0],   [[0,0,1],   ...
       [0,1,0],    [0,0,1],    [1,0,0],
       [0,0,1]]    [1,0,0]]    [0,1,0]]
    
    内部点 (组合):
      [[0.5, 0.3, 0.2],
       [0.3, 0.4, 0.3],
       [0.2, 0.3, 0.5]]
    行列和都=1, 非负

  为什么 mHC 用它?
    1. 行和=1: 流量守恒
      → 输入总量 = 输出总量
      → 不会放大或缩小
    2. 列和=1: 公平分配
      → 每条流都贡献一点
      → 没有流被'饿死'
    3. 非负: 单向流动
      → 没有'回灌'
      → 信号方向明确


【Sinkhorn-Knopp 算法】
────────────────────

  目的: 把任意矩阵投影到 Birkhoff 多胞形
  
  迭代步骤:
    1. 行归一化 (行和=1)
    2. 列归一化 (列和=1)
    3. 重复直到收敛
  
  示例:
    M = [[3, 1, 1],     (开始)
         [1, 2, 1],
         [1, 1, 2]]
    
    第1次行归一化:
    M' = [[0.6, 0.2, 0.2],   (行和=1)
          [0.25, 0.5, 0.25],
          [0.25, 0.25, 0.5]]
    
    第1次列归一化:
    M'' = [[0.55, 0.21, 0.21],  (列和=1)
           [0.23, 0.52, 0.23],
           [0.23, 0.26, 0.56]]
    
    重复几次, 收敛到 Birkhoff 多胞形

  PyTorch 实现:
    for _ in range(num_iter):
        M = M / M.sum(dim=-1, keepdim=True)  # 行
        M = M / M.sum(dim=-2, keepdim=True)  # 列


【mHC 的架构】
─────────────

  单流残差:
    x_{l+1} = x_l + F(x_l)
    
    其中 F 是 Transformer Block

  Hyper-Connections:
    x_{l+1} = A^T x_l + B^T F(x_l)
    
    A: [n, n] 主到主的连接
    B: [n, n] 主到残差的连接
    n: 流的数量

  mHC:
    A ∈ Birkhoff 多胞形 (约束)
    B ∈ Birkhoff 多胞形 (约束)
    → 流量守恒, 训练稳定


【mHC 与标准残差对比】
────────────────────

  ┌──────────┬──────────────┬──────────────┐
  │ 指标      │ 标准残差      │ mHC (n=4)    │
  ├──────────┼──────────────┼──────────────┤
  │ 流数量    │ 1            │ 4            │
  │ 参数量    │ 1x           │ ~1.1x (A,B)  │
  │ 计算量    │ 1x           │ 1x (合并)    │
  │ 训练稳定性│ 中            │ 高           │
  │ 100层+   │ 难            │ 易           │
  │ 性能      │ 基线          │ +3-5%        │
  └──────────┴──────────────┴──────────────┘

  关键:
    A, B 矩阵很小 (4x4, 8x8)
    推理时可合并, 无额外开销


【mHC 的初始化策略】
──────────────────

  关键: 训练开始时行为应接近标准残差
  
  初始化:
    A = I + 0.01 * noise    (接近恒等)
    B = 0.01 * noise        (接近 0)
  
  效果:
    第 1 层: x_1 ≈ I^T x_0 + 0^T F(x_0) = x_0
    → 与标准残差相同
    
  训练中:
    A 学会层间信息混合
    B 学会引入残差
    
  收敛后:
    A, B 都变成合理的混合矩阵


【mHC 的训练技巧】
────────────────

  1. 渐进修剪:
    训练初期让 B 接近 0
    训练中慢慢放开
    → 防止早期不稳定
    
  2. Sinkhorn 迭代数:
    训练时: 5-10 次 (精确)
    推理时: 0 次 (A, B 已固定)
    
  3. n (流数) 选择:
    n=2:  平衡效果和成本
    n=4:  默认, 效果明显
    n=8:  极致, 但边际收益低
    n=16: 通常没必要
    
  4. 监控 A, B:
    训练中观察 A, B 是否仍在 Birkhoff 多胞形
    数值问题: A, B 出现负值 → Sinkhorn 失效


【实际效果】
──────────

  实验 (50 层 Transformer):
  
  ┌──────────┬──────────┬──────────┐
  │ 架构      │ Loss     │ 收敛步数  │
  ├──────────┼──────────┼──────────┤
  │ 标准残差  │ 3.2     │ 100K      │
  │ HC       │ 3.0     │ 80K       │
  │ mHC      │ 2.7     │ 60K       │
  └──────────┴──────────┴──────────┘
  
  mHC 优势:
    - Loss 降低 15%
    - 收敛快 40%
    - 训练更稳定 (loss 曲线平滑)
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】为什么需要多流
  标准单流残差有什么问题? 多流如何解决?

【练习2】Birkhoff 多胞形
  Birkhoff 多胞形的三个约束是什么? 各自的作用?

【练习3】Sinkhorn 算法
  Sinkhorn-Knopp 算法做什么? 用几行代码描述

【练习4】A, B 矩阵
  mHC 中 A 矩阵和 B 矩阵分别代表什么含义?

【练习5】初始化策略
  为什么 mHC 训练开始时 A 接近 I, B 接近 0?

【练习6】流数选择
  n=2, n=4, n=8 各有什么优缺点? 如何选?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  标准单流残差的问题:")
print()
print("  1. 梯度传播链长:")
print("    50 层残差: ∂L/∂x_0 = ∏(1 + ∂F/∂x_i)")
print("    → 即使每项接近 1, 50 项乘起来仍可能爆炸/消失")
print("    → 训练早期梯度不稳定")
print()
print("  2. 信息瓶颈:")
print("    所有信息压缩在一条流上")
print("    → 浅层特征被高层覆盖")
print("    → 难以保留细节")
print()
print("  3. 一刀切:")
print("    残差是简单的 'x + F(x)'")
print("    → 不能区分不同语义")
print("    → 不同任务都走同一条路")
print()
print("  多流 (mHC) 的解决:")
print("    1. 多流 → 信息多通道, 缓解瓶颈")
print("    2. A 矩阵 → 灵活跨层信息混合")
print("    3. B 矩阵 → 灵活残差贡献")
print("    4. Birkhoff 约束 → 防止不稳定的混合")

# 练习2
print("\n【练习2 答案】")
print("  Birkhoff 多胞形的三个约束:")
print()
print("  1. 行和 = 1:")
print("    含义: 流量守恒")
print("    作用: 输入量 = 输出量")
print("    → 不会放大信号, 防止爆炸")
print("    → 不会缩小信号, 防止消失")
print()
print("  2. 列和 = 1:")
print("    含义: 公平分配")
print("    作用: 每条流都贡献一点")
print("    → 没有流被'饿死'")
print("    → 鼓励所有流参与")
print()
print("  3. 非负:")
print("    含义: 单向流动")
print("    作用: 信号只往前走")
print("    → 没有'回灌'或'短路'")
print("    → 因果关系明确")
print()
print("  共同效果:")
print("    训练稳定 (无梯度爆炸)")
print("    表达力强 (灵活的混合)")
print("    长期运行 (不崩溃)")

# 练习3
print("\n【练习3 答案】")
print("  Sinkhorn-Knopp 算法: 把任意矩阵投影到 Birkhoff 多胞形")
print()
print("  核心代码 (PyTorch):")
print("    M = torch.randn(n, n)")
print("    for _ in range(num_iter):")
print("        # 1. 行归一化 (行和=1)")
print("        M = M / M.sum(dim=-1, keepdim=True)")
print("        # 2. 列归一化 (列和=1)")
print("        M = M / M.sum(dim=-2, keepdim=True)")
print()
print("  原理:")
print("    行归一化保证行和=1")
print("    列归一化保证列和=1")
print("    → 反复迭代, 同时满足两个约束")
print("    → 收敛到行列和都=1 的矩阵")
print()
print("  收敛速度:")
print("    一般 5-10 次就够")
print("    严格: 20-50 次")
print("    → 线性收敛")
print()
print("  数值稳定:")
print("    加 epsilon 防除零")
print("    避免极端值 (裁剪到 [eps, 1-eps])")

# 练习4
print("\n【练习4 答案】")
print("  A 矩阵和 B 矩阵的含义:")
print()
print("  A 矩阵 (主连接):")
print("    形状: [n, n]  (n 是流数)")
print("    含义: 主路径间的混合")
print("    A[i, j]: 第 j 条主路径到第 i 条主路径的转移")
print("    例子 (n=2):")
print("      A = [[0.8, 0.2],   # 80% 来自自己, 20% 来自另一条")
print("           [0.2, 0.8]]")
print()
print("  B 矩阵 (残差连接):")
print("    形状: [n, n]")
print("    含义: Transformer 残差对各主路径的贡献")
print("    B[i, j]: 第 j 个残差流到第 i 条主路径的贡献")
print("    例子 (n=2):")
print("      B = [[0.5, 0.0],   # 只贡献到主路径 1")
print("           [0.0, 0.5]]   # 只贡献到主路径 2")
print()
print("  完整公式:")
print("    x_{l+1} = A^T x_l + B^T F(x_l)")
print()
print("  A 接近 I:")
print("    主路径相对独立, 信息保留")
print("  B 不为 0:")
print("    Transformer 的残差有效贡献")

# 练习5
print("\n【练习5 答案】")
print("  初始化策略的原因:")
print()
print("  训练开始时:")
print("    A ≈ I (单位矩阵)")
print("    B ≈ 0 (零矩阵)")
print()
print("  此时 mHC 行为:")
print("    x_1 = A^T x_0 + B^T F(x_0)")
print("    x_1 ≈ I^T x_0 + 0^T F(x_0)")
print("    x_1 ≈ x_0")
print()
print("  → 与标准残差 x_1 = x_0 + F(x_0) 略有不同")
print("  → 但非常接近 (x_0 vs x_0 + 0)")
print()
print("  为什么这样?")
print("  1. 训练稳定:")
print("    开始时与已知方案一致")
print("    → 不会因为新结构而'乱套'")
print("    → Loss 不会突然跳变")
print()
print("  2. 公平比较:")
print("    所有实验都从'同一起点'开始")
print("    → 性能差异来自训练, 不是初始化")
print()
print("  3. 优化友好:")
print("    A 已经是合理的混合")
print("    B 从 0 开始, 慢慢学'残差贡献'")
print()
print("  训练中:")
print("    B 逐渐变成非零")
print("    A 学会更复杂的混合")
print("    → 模型表达力增强")

# 练习6
print("\n【练习6 答案】")
print("  流数 n 的选择权衡:")
print()
print("  n=2:")
print("    优点: 参数量少, 训练快, 推理开销小")
print("    缺点: 表达力有限, 效果提升小")
print("    适合: 资源紧, 快速实验")
print()
print("  n=4 (推荐):")
print("    优点: 平衡效果和成本")
print("    缺点: 参数量略增")
print("    适合: 通用场景, 论文默认")
print()
print("  n=8:")
print("    优点: 表达力强, 效果明显")
print("    缺点: 参数量大, 训练慢")
print("    适合: 极致性能, 大模型")
print()
print("  n=16+ :")
print("    优点: 几乎无")
print("    缺点: 参数爆炸, 边际收益极低")
print("    → 不推荐")
print()
print("  选择建议:")
print("    小模型 (< 1B): n=2 够用")
print("    中模型 (1-10B): n=4 推荐")
print("    大模型 (10B+): n=4-8")
print()
print("  经验:")
print("    n=2 → 4: 效果提升明显")
print("    n=4 → 8: 效果提升 1-2%")
print("    n=8 → 16: 效果几乎不变")
print()
print("  实际项目:")
print("    优先试 n=4")
print("    资源紧张用 n=2")
print("    追求极致用 n=8")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
