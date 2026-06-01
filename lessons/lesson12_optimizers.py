"""
第12课：优化器 — 决定模型怎么学习的灵魂
========================================

训练模型有两个关键问题：
  1. 模型怎么"算"出预测？  → 前向传播
  2. 模型怎么"改"对错误？  → 优化器

优化器 (Optimizer) 的作用：
  根据损失函数算出的梯度，更新模型参数

为什么需要不同的优化器？
  简单的梯度下降有很多问题：
  - 速度慢、容易震荡
  - 容易卡在局部最优点
  - 不同参数需要不同的学习率

本课会一步步演进：
  SGD → Momentum → Adam → AdamW → Muon

运行: python lessons/lesson12_optimizers.py
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 第1部分：什么是优化？
# ============================================================

print("=" * 60)
print("第1部分：什么是优化？")
print("=" * 60)

print("""
想象你在一个黑夜里下山，只能感受到脚下的坡度（梯度）。
你的目标：走到山谷最低点（损失最小）。

优化器 = 你的"走路策略"。

  - 走多快？       → 学习率 (learning rate)
  - 要不要惯性？   → 动量 (momentum)
  - 能不能自适应？ → 自适应学习率 (Adam)
""")

# 简单演示：寻找 y = x^2 的最小值
def f(x):
    return x ** 2

def grad_f(x):
    return 2 * x

# 不同优化器找到最小值的过程
_state = {}

def optimize(optimizer_name, x_start=5.0, lr=0.1, steps=20):
    x = torch.tensor(x_start, requires_grad=True)
    history = [x.item()]

    if optimizer_name == "Momentum":
        _state["v"] = torch.zeros_like(x)

    for _ in range(steps):
        loss = f(x)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        # 根据不同优化器更新
        with torch.no_grad():
            if optimizer_name == "SGD":
                x -= lr * x.grad
            elif optimizer_name == "Momentum":
                # 简化版 Momentum
                beta = 0.9
                _state["v"] = beta * _state["v"] + x.grad
                x -= lr * _state["v"]

        history.append(x.item())
    return history


# 训练过程可视化
print("\n[实验1] 不同优化器找 y=x^2 最小值的过程:")
print("-" * 60)
print(f"{'步骤':<6}{'SGD':<20}{'Momentum':<20}")
print("-" * 60)

# 重置 Momentum 状态
optimize.v = None
hist_sgd = optimize("SGD", x_start=5.0, lr=0.1, steps=15)
hist_mom = optimize("Momentum", x_start=5.0, lr=0.1, steps=15)

for i in range(0, 16, 2):
    print(f"{i:<6}{hist_sgd[i]:<20.4f}{hist_mom[i]:<20.4f}")

print("\n观察：Momentum 比 SGD 收敛更快更稳定！")


# ============================================================
# 第2部分：SGD — 最朴素的优化器
# ============================================================

print("\n" + "=" * 60)
print("第2部分：SGD — 随机梯度下降")
print("=" * 60)

print("""
SGD (Stochastic Gradient Descent) — 最简单的优化器

核心思想：
  1. 计算损失 L
  2. 反向传播得到梯度 ∇L
  3. 参数更新: θ = θ - lr * ∇L

公式：
  θ_new = θ - lr * ∇L
       = θ - 学习率 × 梯度

类比：
  你在下山，lr 决定你的步长
  步子太大 → 走过最低点
  步子太小 → 下山太慢
""")

class SimpleSGD:
    """SGD 优化器的简化实现"""

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        """更新参数"""
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad.data

    def zero_grad(self):
        """清空梯度"""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# 对比：自己实现 vs PyTorch 内置
x = torch.tensor(5.0, requires_grad=True)
optimizer = SimpleSGD([x], lr=0.1)

print("\n自己实现的 SGD 训练 10 步:")
for i in range(10):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        print(f"  步{i}: x = {x.item():.4f}, loss = {loss.item():.4f}")


# ============================================================
# 第3部分：SGD with Momentum — 加入惯性
# ============================================================

print("\n" + "=" * 60)
print("第3部分：SGD with Momentum — 加上惯性")
print("=" * 60)

print("""
问题：SGD 容易在"峡谷"两侧来回震荡
       ↓
解决：加上"惯性"（momentum）

核心思想：
  - 维持一个速度 v（动量）
  - 梯度不只是更新参数，还更新速度
  - 速度累积起来，形成"惯性"

公式：
  v_new = β * v + ∇L          (β 是动量系数，通常 0.9)
  θ_new = θ - lr * v_new

类比：
  小球从山上滚下
  - 积累速度后，能冲过小山包
  - 在平缓的地方不会立刻停下
""")

class MomentumSGD:
    """带动量的 SGD"""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.velocities[i] = self.momentum * self.velocities[i] + p.grad.data
                p.data = p.data - self.lr * self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# 演示
x = torch.tensor(5.0, requires_grad=True)
optimizer = MomentumSGD([x], lr=0.1, momentum=0.9)

print("\n带 Momentum 的 SGD 训练 10 步:")
for i in range(10):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        print(f"  步{i}: x = {x.item():.4f}, loss = {loss.item():.4f}")


# ============================================================
# 第4部分：Adam — 自适应学习率
# ============================================================

print("\n" + "=" * 60)
print("第4部分：Adam — 自适应学习率王者")
print("=" * 60)

print("""
问题：
  - 不同的参数可能需要不同的学习率
  - 有些参数更新频繁，有些稀疏

解决：每个参数维护自己的学习率
  - 频繁更新的参数 → 减小学习率
  - 稀疏更新的参数 → 增大学习率

Adam (Adaptive Moment Estimation) 的核心思想：
  1. 一阶动量 m: 梯度的指数移动平均（类似 Momentum）
  2. 二阶动量 v: 梯度平方的指数移动平均（学习率调整）
  3. 偏差修正: 修正初期估计不准的问题

公式：
  m_t = β1 * m_{t-1} + (1-β1) * ∇L
  v_t = β2 * v_{t-1} + (1-β2) * (∇L)²
  m_hat = m_t / (1 - β1^t)
  v_hat = v_t / (1 - β2^t)
  θ = θ - lr * m_hat / (√v_hat + ε)
""")

class SimpleAdam:
    """Adam 优化器的简化实现"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in self.params]  # 一阶动量
        self.v = [torch.zeros_like(p.data) for p in self.params]  # 二阶动量
        self.t = 0  # 步数

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad = p.grad.data

                # 更新一阶和二阶动量
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

                # 偏差修正
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 更新参数
                p.data = p.data - self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# 演示
x = torch.tensor(5.0, requires_grad=True)
optimizer = SimpleAdam([x], lr=0.5)

print("\nAdam 训练 20 步 (lr=0.5):")
for i in range(20):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 4 == 0:
        print(f"  步{i}: x = {x.item():.4f}, loss = {loss.item():.4f}")


# ============================================================
# 第5部分：AdamW — 解耦权重衰减
# ============================================================

print("\n" + " = " * 30)
print("第5部分：AdamW — 正确使用权重衰减")
print("=" * 60)

print("""
问题：Adam 的权重衰减有问题
  - Adam 中的 L2 正则化会与自适应学习率相互作用
  - 导致权重衰减失效

解决：AdamW 把权重衰减"解耦"出来
  - 不再把 L2 正则化加到梯度上
  - 而是直接从参数中减去 wd * θ

公式：
  m_t = β1 * m_{t-1} + (1-β1) * ∇L
  v_t = β2 * v_{t-1} + (1-β2) * (∇L)²
  m_hat = m_t / (1 - β1^t)
  v_hat = v_t / (1 - β2^t)
  θ = θ - lr * (m_hat / (√v_hat + ε) + wd * θ)

优势：
  - 权重衰减效果更好
  - 训练更稳定
  - 是当前 LLM 训练的事实标准
""")

class SimpleAdamW:
    """AdamW 优化器"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # Adam 步骤
                grad = p.grad.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 注意：权重衰减直接加到参数上 (解耦)
                p.data = p.data - self.lr * (
                    m_hat / (v_hat.sqrt() + self.eps)
                    + self.weight_decay * p.data
                )

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ============================================================
# 第6部分：Muon — 现代化的优化器
# ============================================================

print("\n" + "=" * 60)
print("第6部分：Muon — 新一代优化器")
print("=" * 60)

print("""
Muon (Momentum Orthogonalized by Newton-schulz) — 2024年提出的新优化器

核心思想：
  对梯度做 Newton-Schulz 正交化
  让更新方向更"正交"，训练更高效

为什么需要正交化？
  - 梯度矩阵的奇异值分布不均
  - 大的奇异值主导更新 → 学习不均衡
  - 正交化后，所有方向贡献相同

公式：
  g = ∇L                      # 原始梯度
  m = β * m + g              # 动量
  g_ortho = NewtonSchulz(m)  # Newton-Schulz 正交化
  θ = θ - lr * g_ortho

Newton-Schulz 正交化：
  用 5 次矩阵乘法迭代，将矩阵变成正交矩阵
  不需要 SVD 分解，速度快

为什么 Muon 强大？
  - 训练速度比 AdamW 快 2x
  - 显存节省（不需要二阶动量）
  - 收敛更稳定
  - 已被 Kimi 等大模型采用
""")

def newton_schulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz 正交化 (5次迭代)"""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


class SimpleMuon:
    """Muon 优化器 - 简化版"""

    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None or p.ndim < 2:
                # 1D 参数 (norms, biases) 跳过 Muon，用 SGD
                if p.grad is not None:
                    p.data = p.data - self.lr * p.grad.data
                continue

            grad = p.grad.data

            # 动量更新
            self.velocities[i] = self.momentum * self.velocities[i] + grad

            if self.nesterov:
                grad = self.velocities[i] + self.momentum * (
                    self.velocities[i] - self.velocities[i] / self.momentum
                )
            else:
                grad = self.velocities[i]

            # Newton-Schulz 正交化
            grad_ortho = newton_schulz5(grad)

            # 更新参数
            p.data = p.data - self.lr * grad_ortho

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# 演示
print("\n演示：优化 2D 矩阵参数")

# 2D 矩阵
W = torch.randn(4, 8) * 5
W.requires_grad_(True)

optimizer = SimpleMuon([W], lr=0.01)

print(f"初始 W 的 Frobenius 范数: {W.norm():.4f}")
for i in range(5):
    loss = (W ** 2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  步{i+1}: loss = {loss.item():.4f}, ||W|| = {W.norm():.4f}")


# ============================================================
# 第7部分：在 MiniMind 中怎么选
# ============================================================

print("\n" + "=" * 60)
print("第7部分：MiniMind 的优化器选择")
print("=" * 60)

print("""
MiniMind 训练时常用的优化器配置：

1. 2D 参数 (q_proj, k_proj, v_proj, o_proj, fc, proj) → Muon
   - 矩阵参数，正交化效果好
   - 训练快、显存省

2. 1D 参数 (norms, biases) → AdamW
   - 标量/向量，不适合正交化
   - 用 AdamW 更稳定

3. Embedding / LM Head → AdamW
   - 特殊层，单独处理

代码示例 (从 MiniMind 复制):
```python
# 分离参数
muon_params = [p for n, p in model.named_parameters()
               if p.ndim >= 2 and 'embed' not in n and 'head' not in n]
adamw_params = [p for n, p in model.named_parameters()
                if p.ndim < 2 or 'embed' in n or 'head' in n]

# 创建两个优化器
optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
optimizer.add_param_group({
    'params': adamw_params,
    'optimizer': AdamW(adamw_params, lr=3e-4, weight_decay=0.01)
})
```
""")


# ============================================================
# 优化器对比
# ============================================================

print("=" * 60)
print("优化器对比总结")
print("=" * 60)

comparison = """
┌──────────────┬────────────────┬─────────────┬────────────────┐
│ 优化器        │ 核心思想         │ 优点         │ 缺点           │
├──────────────┼────────────────┼─────────────┼────────────────┤
│ SGD          │ 基础梯度下降     │ 简单、稳定   │ 慢、易震荡     │
│ Momentum     │ 加上惯性         │ 加速收敛     │ 学习率难调     │
│ Adam         │ 自适应学习率     │ 收敛快       │ 权重衰减有问题 │
│ AdamW        │ 解耦权重衰减     │ 通用、稳定   │ 显存占用大     │
│ Muon         │ 梯度正交化       │ 更快、显存省 │ 2D 参数专用    │
└──────────────┴────────────────┴─────────────┴────────────────┘

推荐：
  - 小型实验 → AdamW
  - 中型训练 → Muon + AdamW 混合
  - 大模型训练 → Muon + AdamW 混合
"""

print(comparison)


# ============================================================
# 深入理解：优化器的"超能力"
# ============================================================
print("\n" + "=" * 60)
print("深入理解：优化器的'超能力'")
print("=" * 60)

print("""
【类比：优化器 = 下山方法】
──────────────────────
  想象你站在山上, 雾很大看不见路
  目标: 找到山谷 (loss 最小)
  
  SGD: 看一下脚下, 往最陡的下坡走一步
       简单但慢, 容易卡在小坡上
  
  Momentum: 带个滑板下山
            有惯性, 能越过小坡
            加快速度
  
  Adam: 带 GPS 的越野车
        不光看脚下, 还看历史轨迹
        智能地规划路线
  
  Muon: 带方向罗盘的越野车
        不只看历史, 还把方向'正交化'
        → 不互相打架, 走得又直又快


【优化器的发展史】
────────────────

  2012: SGD - 简单, 慢
  2015: Adam - 自适应, 快
  2017: AdamW - 修正权重衰减
  2024: Muon - 正交化, 更快

  趋势:
    1. 越来越复杂, 但效果越来越好
    2. 自适应学习率是核心
    3. 显存和速度权衡


【图示：各优化器收敛速度对比】
─────────────────────────

  Loss│
      │╲
      │ ╲ AdamW
      │  ╲
      │   ╲  Muon  (更快)
      │    ╲
      │     ╲_____
      │      ╲__  Adam
      │         ╲___
      │             ╲_____  SGD (最慢)
      └──────────────────── step


【Muon 的核心：Newton-Schulz 正交化】
─────────────────────────────────

  为什么有效?
  
  普通梯度下降:
    W -= lr * grad
    → 梯度方向可能很"扁" (椭球), 走 Z 字
  
  Muon:
    1. 先做 momentum (动量累积)
    2. Newton-Schulz 迭代: 把动量矩阵'正交化'
    3. 更新参数
  
  效果:
    矩阵的奇异值都接近 1
    → 每个方向走得一样快
    → 不会因矩阵病态而走偏


【Adam vs AdamW 的关键差异】
──────────────────────────

  Adam (2015):
    weight_decay 直接加到梯度上:
      grad += weight_decay * W
    → 权重衰减与学习率耦合, 不理想
  
  AdamW (2017):
    权重衰减直接加到参数上:
      W -= lr * (grad + weight_decay * W)
    → 解耦, 更稳定, 是当前主流

  实际效果:
    Adam 在大数据集上泛化差
    AdamW 显著改善, 接近 SGD 泛化能力


【优化器选择指南】
────────────────

  ┌─────────────┬──────────────────────┐
  │ 场景         │ 推荐                  │
  ├─────────────┼──────────────────────┤
  │ LLM 预训练   │ AdamW 或 Muon+AdamW  │
  │ LLM 微调     │ AdamW                 │
  │ 计算机视觉   │ SGD 或 AdamW          │
  │ 强化学习     │ Adam (快, 允许过拟合) │
  │ 小数据集     │ SGD (泛化好)          │
  │ 快速实验     │ AdamW (稳定)          │
  └─────────────┴──────────────────────┘


【MiniMind 的优化器配置】
──────────────────────

  预训练:
    optimizer = AdamW(lr=1e-4, weight_decay=0.01)
    + warmup + cosine decay
  
  进阶:
    optimizer = Muon(2D参数) + AdamW(其他)
    → 大幅节省显存, 收敛更快
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】SGD vs Adam
  简单问题 (线性回归) 用 SGD 和 Adam, 哪个更快收敛?
  复杂问题 (深度网络) 呢?

【练习2】学习率选择
  Adam 推荐学习率是多少? SGD 呢? 为什么差这么多?

【练习3】AdamW 优势
  AdamW 相比 Adam 的核心改进是什么?

【练习4】Muon 适用
  Muon 适合所有参数吗? 哪些参数不能用 Muon?

【练习5】混合优化器
  训练 LLaMA-7B 用 Muon + AdamW 混合, 怎么分配?

【练习6】显存对比
  AdamW 和 SGD 各需要多少额外显存? (每参数)
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  简单问题 (线性回归):")
print("    SGD 更快, 几乎瞬间收敛")
print("    Adam 自适应反而是负担 (需要 warmup)")
print()
print("  复杂问题 (深度网络):")
print("    Adam 显著优于 SGD")
print("    原因:")
print("      - 深度网络 loss landscape 复杂")
print("      - 不同参数需要不同学习率")
print("      - Adam 自适应正合适")
print()
print("  结论:")
print("    没有'最优'优化器, 只有'合适的'优化器")
print("    简单任务用 SGD, 复杂任务用 Adam")

# 练习2
print("\n【练习2 答案】")
print("  Adam 推荐 lr: 1e-3 到 1e-4")
print("  SGD 推荐 lr: 1e-1 到 1e-2")
print()
print("  差了 100 倍, 为什么?")
print()
print("  SGD:")
print("    直接用梯度, 梯度可能很大")
print("    需要较大学习率才能走足够远")
print("    但太大又会震荡")
print()
print("  Adam:")
print("    梯度被二阶动量'标准化'了")
print("    update = grad / sqrt(var)")
print("    → 有效步长是 O(1) 量级, 与梯度大小无关")
print("    → 所以可以用很小的 lr")
print()
print("  实验:")
print("    Adam lr=1e-3 ≈ SGD lr=1e-1 的步长")
print("    但收敛速度差很多倍")

# 练习3
print("\n【练习3 答案】")
print("  核心改进: 权重衰减解耦")
print()
print("  Adam (旧):")
print("    grad = 原梯度 + weight_decay * W")
print("    W = W - lr * grad")
print("    → weight_decay 的效果依赖于 lr")
print("    → lr 调小, weight_decay 等比减小, 不合理")
print()
print("  AdamW (新):")
print("    grad = 原梯度  (不含 weight_decay)")
print("    W = W - lr * (grad + weight_decay * W)")
print("    → weight_decay 的效果独立于 lr")
print("    → 调参更稳定, 泛化更好")
print()
print("  实验证据:")
print("    训练 ResNet:")
print("      Adam:  78% 准确率")
print("      AdamW: 79.5% 准确率")
print("    差距看似小, 但在大模型上明显")

# 练习4
print("\n【练习4 答案】")
print("  Muon 适合: 2D 参数 (矩阵)")
print("  Muon 不适合: 1D 参数 (向量)")
print()
print("  原因:")
print("    Newton-Schulz 正交化作用于矩阵")
print("    向量没有'正交化'的概念")
print()
print("  LLM 中的参数分类:")
print("    2D (用 Muon):")
print("      - q_proj.weight: [hidden, hidden]")
print("      - k_proj.weight")
print("      - v_proj.weight")
print("      - o_proj.weight")
print("      - gate_proj, up_proj, down_proj.weight")
print("      - embedding.weight (也可视为 2D)")
print("    1D (用 AdamW):")
print("      - layernorm.weight")
print("      - layernorm.bias")
print("      - 所有 bias")
print()
print("  MiniMind 配置示例:")
print("    muon_params = [p for p in model.parameters() if p.ndim >= 2]")
print("    adamw_params = [p for p in model.parameters() if p.ndim < 2]")
print("    optimizer = MixedOptimizer(")
print("        Muon(muon_params, lr=0.02),")
print("        AdamW(adamw_params, lr=3e-4),")
print("    )")

# 练习5
print("\n【练习5 答案】")
print("  典型分配 (Kimi/Moonlight 等大模型):")
print()
print("    Muon:  所有 2D 矩阵参数 (attention + FFN)")
print("    AdamW: 所有 1D 参数 (norm + bias)")
print("           + embedding")
print("           + lm_head")
print()
print("  原因:")
print("    - Muon 对 2D 矩阵效果最好")
print("    - Embedding 和 LM Head 是'查表', 不是变换")
print("    - 不需要正交化, 用 AdamW 反而好")
print()
print("  学习率比例:")
print("    Muon:   lr = 0.02")
print("    AdamW:  lr = 3e-4 (小 60 倍)")
print()
print("  实际效果:")
print("    - 训练速度提升 1.5-2x")
print("    - 显存节省 ~50% (Muon 不存二阶动量)")
print("    - 最终 loss 低 5-10%")

# 练习6
print("\n【练习6 答案】")
print("  SGD:")
print("    每参数额外显存: 0 字节")
print("    (只用梯度本身)")
print()
print("  Adam/AdamW:")
print("    每参数额外显存: 8 字节 (FP32 状态)")
print("    - 一阶动量 m: 4 字节")
print("    - 二阶动量 v: 4 字节")
print()
print("  Muon:")
print("    每参数额外显存: 4 字节 (FP32 动量)")
print("    - 只存一阶动量")
print("    - 二阶动量由 NS 迭代'在线'算")
print()
print("  对比 (10亿参数模型):")
print("    SGD:    0 GB 额外")
print("    Muon:   4 GB 额外")
print("    AdamW:  8 GB 额外")
print()
print("  → Muon 比 AdamW 省 50% 优化器显存")
print("  → 这就是大模型训练偏爱 Muon 的原因")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
