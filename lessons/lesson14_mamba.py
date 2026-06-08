"""
第14课：Mamba — 颠覆 Attention 的新架构
=======================================

Mamba (Albert Gu & Tri Dao, 2023) 是 Transformer 的有力挑战者。

核心创新：
  1. 状态空间模型 (SSM) 作为基础
  2. 选择性机制 (Selective SSM) — 让模型"筛选"重要信息
  3. 线性复杂度 O(N) — 完胜 Attention 的 O(N²)

本课会讲解：
  1. 什么是状态空间模型
  2. 选择性扫描的工作原理
  3. Mamba Block 的完整结构
  4. 在 MiniMind 中怎么用

运行: python lessons/lesson14_mamba.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 第1部分：什么是状态空间模型 (SSM)
# ============================================================

print("=" * 60)
print("第1部分：什么是状态空间模型 (SSM)")
print("=" * 60)

print("""
状态空间模型 (State Space Model) 来自控制论：
  描述一个系统如何随时间演化

连续形式：
  h'(t) = A·h(t) + B·x(t)   # 状态更新
  y(t)  = C·h(t) + D·x(t)   # 输出

其中:
  x(t) = 输入
  h(t) = 隐藏状态 (压缩了所有历史信息)
  y(t) = 输出
  A, B, C, D = 系统参数

类比：
  x(t) = 不断流进来的水
  h(t) = 水池
  y(t) = 流出去的水
  A = 水池的"漏水速度"
  B = 进水阀门
  C = 出水阀门

为什么 SSM 适合序列建模？
  - 隐藏状态 h(t) 可以压缩任意长历史 → 固定大小！
  - 计算效率高
  - 训练时可以并行 (用卷积表示)
  - 推理时是 O(1) 每步

【SSM vs Attention：为什么 SSM 可以替代 Attention？】

  Attention 的做法（第5课学过的）：
    每个位置都要看所有其他位置 → O(N²) 复杂度
    "你好世界" → 每个字都要和所有字算注意力
    优点：能直接看到任意远处的信息
    缺点：序列越长越慢，内存越大

  SSM 的做法：
    每个位置只更新一个固定大小的状态 → O(N) 复杂度
    "你好世界" → 每个字更新状态 h，下一个字只看 h
    优点：序列多长都一样快
    缺点：信息压缩到状态 h 中，可能丢失细节

  直觉类比：
    Attention = 开会时每个人都能直接和所有人对话（信息丰富但慢）
    SSM = 开会时每个人只听前一个人的总结（快速但信息压缩）

    想象你在读一本书：
    Attention = 每读一个字，都翻回前面所有页重新看一遍（精确但慢）
    SSM = 每读一个字，只更新你脑中的"理解"（快速但可能遗漏细节）

  Mamba 的创新 = SSM + 选择性机制：
    普通SSM：所有信息都同等对待（像流水线，不筛选）
    Mamba：重要信息多记住，不重要信息少记住（像有注意力的流水线）
    → 兼顾了 SSM 的速度和 Attention 的选择性
""")


# ============================================================
# 第2部分：S4 — 高效的 SSM 实现
# ============================================================

print("\n" + "=" * 60)
print("第2部分：S4 — 训练时用卷积, 推理时用循环")
print("=" * 60)

print("""
S4 (Structured State Space) 的关键洞察:
  SSM 可以用两种方式计算：
    1. 循环形式（适合推理）
       h_t = A·h_{t-1} + B·x_t
       y_t = C·h_t

    2. 卷积形式（适合训练）
       y = x * K    (其中 K = (C·B, C·A·B, C·A²·B, ...))

  训练时：把 K 算出来，用卷积并行算
  推理时：转回循环形式，O(1) 每步
""")


# 简单的循环 SSM 演示
def simple_ssm_recurrent(A, B, C, x):
    """循环形式 SSM - 适合推理 (1D 简化版本)"""
    seq_len = x.shape[0]
    h = torch.tensor(0.0)
    outputs = []
    for t in range(seq_len):
        h = A * h + B * x[t]  # 状态更新
        y = C * h            # 输出
        outputs.append(y)
    return torch.stack(outputs)


# 简单的卷积 SSM 演示
def simple_ssm_conv(A, B, C, x):
    """卷积形式 SSM - 适合训练 (1D 简化版本)"""
    seq_len = x.shape[0]
    K = []  # 卷积核
    A_pow = torch.tensor(1.0)
    for t in range(seq_len):
        K.append((C * B * A_pow).item())
        A_pow = A_pow * A
    K = torch.tensor(K)
    # 用 1D 卷积
    y = torch.nn.functional.conv1d(
        x.view(1, 1, -1), K.view(1, 1, -1)
    ).view(-1)
    return y


print("\n[演示] 循环 vs 卷积 SSM 数值一致")
print("-" * 60)
A = torch.tensor(0.8)
B = torch.tensor(0.3)
C = torch.tensor(0.5)
x = torch.randn(10)

y_recurrent = simple_ssm_recurrent(A, B, C, x)
y_conv = simple_ssm_conv(A, B, C, x)

print(f"循环形式输出: {[f'{v:.3f}' for v in y_recurrent[:5].tolist()]}")
print(f"卷积形式输出: {[f'{v:.3f}' for v in y_conv[:5].tolist()]}")
print(f"最大差异: {(y_recurrent - y_conv).abs().max():.2e}")


# ============================================================
# 第3部分：Mamba 的核心创新 — 选择性机制
# ============================================================

print("\n" + "=" * 60)
print("第3部分：Mamba 的核心创新 — 选择性机制")
print("=" * 60)

print("""
传统 SSM 的问题：
  A, B, C, D 是固定的，与输入无关
  → 模型不能根据输入"动态调整"
  → 不能选择性记住或忘记

Mamba 的解决方案：让 B, C, Δ 依赖输入！
  B, C, Δ = Linear(x)   # 由输入动态生成

  Δ (delta) 是关键创新 — 它控制"时间步长"
  - Δ 大：当前输入重要, 快速更新状态
  - Δ 小：当前输入不重要, 保持历史

公式：
  B_t = Linear_B(x_t)
  C_t = Linear_C(x_t)
  Δ_t = softplus(Linear_Δ(x_t))   # 保证 > 0
  h_t = exp(A · Δ_t) · h_{t-1} + Δ_t · B_t · x_t
  y_t = C_t · h_t

类比：
  传统 SSM = 固定大小的水池，固定流速
  Mamba    = 智能水池，水流速度随内容调整
              - 重要信息 → 快速流入
              - 不重要 → 慢慢流入
""")


# ============================================================
# 第4部分：Mamba Block 完整实现
# ============================================================

print("\n" + "=" * 60)
print("第4部分：Mamba Block 完整实现")
print("=" * 60)


class MambaBlock(nn.Module):
    """简化版 Mamba Block - 展示核心结构"""

    def __init__(self, hidden_size, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = hidden_size * expand

        # 归一化
        self.norm = nn.LayerNorm(hidden_size)

        # 输入投影（扩展维度）
        self.in_proj = nn.Linear(hidden_size, self.d_inner * 2, bias=False)

        # 1D 深度可分离卷积（捕获局部信息）
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )

        # SSM 参数
        # A: (d_inner, d_state) — 对数空间参数化（保证负值）
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).expand(self.d_inner, -1))
        # D: 跳跃连接
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 动态参数 B, C, Δ 的投影
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # 输出投影
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)

        # 输入投影，拆成两路
        xz = self.in_proj(x)  # (b, s, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 各 (b, s, d_inner)

        # 1D 卷积
        x = x.transpose(1, 2)  # (b, d_inner, s)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # (b, s, d_inner)
        x = F.silu(x)

        # 动态生成 B, C, Δ
        x_proj = self.x_proj(x)  # (b, s, 2*d_state+1)
        B = x_proj[:, :, :self.d_state]
        C = x_proj[:, :, self.d_state:2*self.d_state]
        dt = x_proj[:, :, -1:]  # (b, s, 1)

        # 限制 Δ 范围
        dt = F.softplus(self.dt_proj(dt))  # (b, s, d_inner)

        # SSM 状态空间
        A = -torch.exp(self.A_log)  # (d_inner, d_state) — 负值

        # 选择性扫描（这里用简单循环演示）
        h = torch.zeros(bsz, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(seq_len):
            # 离散化: h_t = exp(A * Δ_t) * h_{t-1} + Δ_t * B_t * x_t
            dA = torch.exp(A * dt[:, t].unsqueeze(-1))  # (b, d_inner, d_state)
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)  # (b, d_inner, d_state)
            h = dA * h + dB * x[:, t].unsqueeze(-1)
            # 输出: y_t = C_t * h_t
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (b, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (b, s, d_inner)

        # 跳跃连接
        y = y + self.D * x

        # 门控
        y = y * F.silu(z)

        # 输出投影 + 残差
        return residual + self.out_proj(y)


# 演示 Mamba Block
print("\n[实验] Mamba Block 形状验证")
print("-" * 60)
mamba = MambaBlock(hidden_size=64, d_state=16)
x = torch.randn(2, 16, 64)
out = mamba(x)
print(f"输入: {x.shape}  →  输出: {out.shape}")
print(f"参数量: {sum(p.numel() for p in mamba.parameters()):,}")


# ============================================================
# 第5部分：Mamba vs Attention 对比
# ============================================================

print("\n" + "=" * 60)
print("第5部分：Mamba vs Attention")
print("=" * 60)

comparison = """
┌──────────────────┬─────────────────┬────────────────┐
│ 维度              │ Attention       │ Mamba          │
├──────────────────┼─────────────────┼────────────────┤
│ 复杂度 (训练)      │ O(N²)           │ O(N log N)     │
│ 复杂度 (推理)      │ O(N²)           │ O(1) 每步      │
│ 显存               │ O(N²)           │ O(N)           │
│ 长上下文           │ 困难            │ 强             │
│ 检索能力           │ 强              │ 弱             │
│ 推理速度           │ 受 KV Cache 限制 │ 极快            │
│ 训练并行性         │ 完全并行         │ 选择性扫描      │
└──────────────────┴─────────────────┴────────────────┘

Mamba 的优势:
  ✓ 推理时 O(1) 每步 (只要维护状态)
  ✓ 显存占用低 (无 KV Cache)
  ✓ 非常适合长序列
  ✓ 训练速度与 Transformer 相当

Mamba 的劣势:
  ✗ 不能精确"检索" (不像 Attention 那样查询)
  ✗ 选择性扫描实现复杂
  ✗ 训练时仍需循环（已通过并行算法部分解决）
  ✗ 对 in-context learning 较弱

Hybrid 方案 (MiniMind 采用):
  底层用 Mamba (快速处理长序列)
  顶层用 Attention (精确建模)
"""

print(comparison)


# ============================================================
# 第6部分：在 MiniMind 中怎么用
# ============================================================

print("\n" + "=" * 60)
print("第6部分：MiniMind 中的 Mamba 混合架构")
print("=" * 60)

print("""
MiniMind 支持 Mamba + Attention 混合架构:

  [Mamba × 3] → [Mamba × 3] → [Attention × 2] → [Attention × 2]
       快速处理局部信息           精确建模全局依赖

配置示例:
```python
config = MiniMindConfig(
    hidden_size=512,
    num_hidden_layers=8,
    mamba_hybrid=True,        # 启用混合架构
    mamba_ratio=0.5,          # 前 50% 用 Mamba
    d_state=16,               # Mamba 状态维度
)
```

代码:
```python
class MiniMindBlock:
    def __init__(self, config, layer_id):
        if config.mamba_hybrid and layer_id < int(config.num_hidden_layers * config.mamba_ratio):
            self.self_attn = MambaLayer(config)  # Mamba
        else:
            self.self_attn = Attention(config)    # 标准 Attention
```
""")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. 状态空间模型 (SSM)
   - 连续微分方程: h'=A·h+B·x, y=C·h
   - 隐藏状态压缩整个历史 → 固定大小
   - 训练用卷积形式，推理用循环形式

2. Mamba 的核心创新
   - 选择性机制：B, C, Δ 依赖输入
   - 让模型能"动态选择"记住什么、忘记什么
   - 硬件友好的并行扫描算法

3. Mamba Block 结构
   - 1D 卷积（局部信息）
   - 选择性 SSM（长程依赖）
   - 门控机制

4. Mamba vs Attention
   - Mamba: O(N) 复杂度, 推理快, 无 KV Cache
   - Attention: O(N²) 复杂度, 检索能力强
   - 混合架构：取两者之长

下一步: 第15课 - LoRA 微调 (高效适配大模型)
""")


# ============================================================
# 深入理解：Mamba 的"选择性记忆"
# ============================================================
print("\n" + "=" * 60)
print("深入理解：Mamba 的'选择性记忆'")
print("=" * 60)

print("""
【类比：Mamba = 智能笔记本】
──────────────────────
  
  Transformer (Attention):
    像开会: 每说一句话都要看所有历史
    → 信息全, 但慢
  
  RNN/LSTM:
    像脑子: 一边说一边记
    → 快, 但记不清细节
  
  Mamba (S6):
    像带筛选器的智能笔记本:
      - 记当前的重要信息
      - 忘记不重要的
      - 必要时快速翻到任意一页
    → 兼具速度和记忆能力

  具体:
    h_t = A * h_{t-1} + B * x_t      ← 更新记忆
    y_t = C * h_t                      ← 读记忆
    
    其中 A, B, C 是输入的函数 (这是关键创新!)


【Mamba 的核心：选择性 SSM (S6)】
────────────────────────────────

  传统 SSM (S4):
    A, B, C 是固定参数
    → 所有输入用同一套记忆
    → 适应性差
  
  Mamba (S6):
    A, B, C 都由输入 x 动态生成
    → 不同的输入用不同的记忆策略
    → 适应性极强
  
  这就是为什么 Mamba 能"选择性"地记/忘:
    - 看到关键词 → 大 B, 强更新
    - 看到废话 → 小 B, 弱更新
    - 长期记忆 → 适当 A, 慢衰减
    - 短期记忆 → 衰减快, 快速清空


【图示：SSM 状态更新】
────────────────────

  时间步 t=1:    t=2:    t=3:    t=4:
  
  x_1 → [B_1]    x_2 → [B_2]    x_3 → [B_3]    x_4 → [B_4]
       ↗ ↓ ↘        ↗ ↓ ↘        ↗ ↓ ↘        ↗ ↓ ↘
  [A_1] [h_1]    [A_2] [h_2]    [A_3] [h_3]    [A_4] [h_4]
       ↘ ↓ ↗        ↘ ↓ ↗        ↘ ↓ ↗        ↘ ↓ ↗
       y_1          y_2          y_3          y_4

  公式:
    h_t = A_t * h_{t-1} + B_t * x_t
    y_t = C_t * h_t
  
  其中 A_t, B_t, C_t 由 x_t 生成 (Mamba 创新)


【Mamba vs Transformer 复杂度】
────────────────────────────

  ┌────────────┬──────────┬──────────┬──────────┐
  │ 指标        │ 训练      │ 推理      │ 显存      │
  ├────────────┼──────────┼──────────┼──────────┤
  │ Transformer│ O(N²)    │ O(N) KV  │ O(N²)    │
  │ Mamba      │ O(N)     │ O(1)/步  │ O(N)     │
  └────────────┴──────────┴──────────┴──────────┘

  训练:
    Transformer: 每个 token 看所有 (N²)
    Mamba:       每个 token 顺序处理 (N)
    → 长序列时 Mamba 快得多

  推理:
    Transformer: 有 KV Cache, 步长 O(1)
    Mamba:       步长 O(1) (无 Cache)
    → 速度相当, 但 Mamba 显存省

  显存:
    Transformer: KV Cache 占 O(N)
    Mamba:       只需当前状态 O(1)
    → Mamba 显存极省

  实际:
    长序列 (16K+): Mamba 优势巨大
    短序列 (<2K): 差别不大


【Mamba 的硬件高效算法】
──────────────────────

  问题: 朴素的循环 O(N) 不能并行
  
  解决: Parallel Scan (并行扫描)
    把 N 步串行变成 O(log N) 步并行
    
  例子: 8 个元素的累加
    串行: 1→2→3→4→5→6→7→8 (8 步)
    并行: (1+2) (3+4) (5+6) (7+8)
         ((1+2)+(3+4)) ((5+6)+(7+8))
         (((1+2)+(3+4))+((5+6)+(7+8)))
         (3 步, 快了 2.6x)
    
  实际加速:
    序列 8K:  ~16x 加速
    序列 16K: ~25x 加速
    序列 32K: ~40x 加速


【Mamba 块 vs Transformer 块】
────────────────────────────

  Transformer 块:
    x → RMSNorm → Attention → Add
        → RMSNorm → FFN       → Add
    → 输出

  Mamba 块:
    x → Norm → in_proj → Conv1d → SSM → out_proj
                ↓
                SiLU (门控)
    → 输出 + 残差

  关键差异:
    - 没有 Attention, 用 SSM
    - 引入卷积 (Conv1d) 捕捉局部
    - 用 SiLU 门控
    - 整体更轻量


【混合架构: Mamba + Attention】
─────────────────────────────

  纯 Mamba 弱在哪:
    - 精确检索能力 (例如"复制某个 token")
    - in-context learning

  纯 Transformer 弱在哪:
    - 长序列效率
    - 推理显存

  混合 (Jamba, Zamba):
    - 部分层用 Mamba
    - 部分层用 Attention
    - 取两者之长

  比例 (经验):
    6-8 层 Mamba + 1 层 Attention
    → 兼具效率和精度

  MiniMind:
    支持配置 mamba_layers / attn_layers 比例
    默认 1:1 混合
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】SSM 的状态维度
  Mamba 状态 h 的维度是多大? 这代表什么?

【练习2】A, B, C 的角色
  SSM 中 A, B, C 各代表什么含义? 类比 RNN 的门控机制

【练习3】Mamba vs LSTM
  Mamba 相比 LSTM 的核心改进是什么?

【练习4】Mamba 没有 KV Cache
  为什么 Mamba 不需要 KV Cache? 这有什么好处?

【练习5】Mamba 长序列优势
  序列 N=32768, Mamba 和 Transformer 显存差多少?
  (Mamba 状态 [B, D, N] vs Transformer KV [B, L, N, H, D])

【练习6】混合架构
  为什么不能全部用 Mamba? 需要 Attention 帮忙?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  Mamba 状态 h 的维度: [batch, d_inner, d_state]")
print("    d_inner:  内部维度 (类似 hidden_size)")
print("    d_state:  状态维度 (默认 16, 较小)")
print()
print("  含义:")
print("    - d_state = 16 看似很小, 但功能强大")
print("    - 每个维度可看作'一个记忆单元'")
print("    - 16 个记忆单元可表示复杂模式")
print()
print("  对比:")
print("    LSTM 状态: 几百到几千维")
print("    Mamba 状态: 16 维 (更紧凑)")
print()
print("  为什么 16 就够?")
print("    - A, B, C 是输入的函数, 自适应")
print("    - 不需要存所有历史, 只需'摘要'")
print("    - 类似于压缩存储")

# 练习2
print("\n【练习2 答案】")
print("  A (状态转移矩阵):")
print("    - 决定旧记忆保留多少")
print("    - A 接近 1: 长期记忆")
print("    - A 接近 0: 快速遗忘")
print("    - 类比 LSTM 的遗忘门")
print()
print("  B (输入映射):")
print("    - 决定新输入的权重")
print("    - B 大: 当前输入重要")
print("    - B 小: 当前输入忽略")
print("    - 类比 LSTM 的输入门")
print()
print("  C (输出映射):")
print("    - 决定如何从状态读出")
print("    - 类似 LSTM 的输出门")
print()
print("  Mamba 的创新:")
print("    A, B, C 都由输入动态生成")
print("    → 比 LSTM 的固定门控更灵活")

# 练习3
print("\n【练习3 答案】")
print("  Mamba 相对 LSTM 的核心改进:")
print()
print("  1. 状态空间 vs 门控循环:")
print("    LSTM: 三个门控, 复杂但难以并行")
print("    Mamba: SSM 形式, 数学优雅")
print()
print("  2. 选择性机制:")
print("    LSTM: 门控是'硬开关' (0/1)")
print("    Mamba: A, B, C 是连续值, 更平滑")
print()
print("  3. 并行训练:")
print("    LSTM: 必须串行, 训练慢")
print("    Mamba: 用 Parallel Scan, 可并行")
print()
print("  4. 长距离依赖:")
print("    LSTM: 几步后就忘了")
print("    Mamba: 状态机制天然保持长距离")
print()
print("  5. 硬件效率:")
print("    LSTM: 难以在 GPU 高效运行")
print("    Mamba: 专为 GPU 设计 (类似 Attention)")
print()
print("  性能:")
print("    同样参数下, Mamba 比 LSTM 强很多")
print("    训练速度: Mamba 快 5-10x")

# 练习4
print("\n【练习4 答案】")
print("  Mamba 不需要 KV Cache 的原因:")
print()
print("  Transformer:")
print("    注意力需要所有历史 K, V")
print("    生成时必须存起来 → KV Cache")
print()
print("  Mamba:")
print("    状态 h_t 已经'压缩'了所有历史")
print("    不需要单独存 K, V")
print("    只需当前 h_t (固定大小)")
print()
print("  好处:")
print("    1. 显存省:")
print("      Transformer: KV Cache 占 O(N)")
print("      Mamba:       状态占 O(1) (与序列长度无关)")
print()
print("    2. 推理快:")
print("      不需要拼接新 K, V")
print("      每次只需更新 h_t")
print()
print("    3. 长序列友好:")
print("      Transformer: 32K 序列要 32K 步的 Cache")
print("      Mamba:       永远只需当前状态")
print()
print("  实际数据:")
print("    32K 序列推理:")
print("      Transformer: 数十 GB Cache")
print("      Mamba:       < 1 MB")

# 练习5
print("\n【练习5 答案】")
N = 32768
batch = 1
d_inner = 1024
d_state = 16
num_layers = 32
num_kv_heads = 8
head_dim = 64
fp16 = 2

# Mamba 状态
mamba_bytes = batch * d_inner * d_state * fp16 * num_layers
mamba_mb = mamba_bytes / (1024 * 1024)

# Transformer KV
tf_bytes = 2 * batch * num_layers * N * num_kv_heads * head_dim * fp16
tf_mb = tf_bytes / (1024 * 1024)
tf_gb = tf_mb / 1024

print(f"  配置: bs={batch}, N={N}, layers={num_layers}")
print(f"  Mamba 状态: [{batch}, {d_inner}, {d_state}] = {mamba_bytes/1024:.1f} KB")
print(f"  每层状态: {batch * d_inner * d_state * fp16} 字节 = {batch * d_inner * d_state * fp16/1024:.1f} KB")
print(f"  Mamba 32 层总状态: {mamba_mb:.1f} MB")
print()
print(f"  Transformer KV:")
print(f"  每层 KV: {2 * batch * N * num_kv_heads * head_dim * fp16/1024/1024:.1f} MB")
print(f"  32 层总 KV: {tf_mb:.1f} MB = {tf_gb:.2f} GB")
print()
print(f"  → 32K 序列, Mamba 比 Transformer 节省 {(1 - mamba_mb/tf_mb)*100:.0f}% 显存")
print(f"  → 这就是 Mamba 适合超长序列的根本原因")

# 练习6
print("\n【练习6 答案】")
print("  不能全用 Mamba 的原因:")
print()
print("  1. 精确检索弱:")
print("    Transformer 的 Attention 可以精确'看'到某位置")
print("    Mamba 是压缩摘要, 难以精确定位")
print("    → 不适合: 代码补全, 引用查找")
print()
print("  2. 复杂推理弱:")
print("    多步推理需要'线索传递'")
print("    Mamba 的状态被压缩, 传递能力差")
print("    → 不适合: 数学推理, 逻辑分析")
print()
print("  3. 训练数据少时:")
print("    Mamba 需要更多数据学'选择性'")
print("    小数据训练, Transformer 更稳")
print()
print("  需要 Attention 的场景:")
print("    - 检索增强 (RAG)")
print("    - 精确复制/引用")
print("    - In-context learning")
print()
print("  混合策略:")
print("    - 偶数层: Mamba (效率)")
print("    - 奇数层: Attention (精度)")
print("    - 比例 4:1 或 6:1")
print()
print("  实际模型:")
print("    Jamba: Mamba + Attention 1:7")
print("    Zamba: 交替使用")
print("    MiniMind: 可配置")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
