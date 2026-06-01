"""
第15课：LoRA — 用极少的参数微调大模型
=======================================

全量微调 (Full Fine-Tuning) 的问题:
  假设模型有 7B 参数 → Adam 优化器需要 3 倍参数 (m, v, grad)
  显存占用 = 4 bytes × 7B × 3 × 2 = 168GB (fp32)
  根本不可能在普通 GPU 上微调！

LoRA (Low-Rank Adaptation) 的解决方案:
  冻结原始权重 W
  注入两个低秩矩阵 A 和 B
  只训练 A 和 B (参数极少！)

本课会讲解:
  1. 什么是低秩分解
  2. LoRA 的数学原理
  3. LoRA 的 PyTorch 实现
  4. 在 MiniMind 中怎么用

运行: python lessons/lesson15_lora.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 第1部分：全量微调的困境
# ============================================================

print("=" * 60)
print("第1部分：全量微调的困境")
print("=" * 60)

print("""
全量微调 (Full Fine-Tuning) 需要更新所有参数。

假设一个 7B 参数的模型:
  7B × 4 bytes (fp32) = 28GB  权重
  7B × 4 bytes = 28GB           梯度
  7B × 4 bytes × 2 = 56GB      Adam 状态 (m, v)
  7B × 4 bytes = 28GB           激活值 (近似)
  ─────────────────
  总计 ≈ 140GB

  顶级 A100 GPU 80GB × 2 也才 160GB！无法训练！

微调 (Fine-Tuning) 的目的:
  适应新任务 (问答、对话、代码、...)
  解决"灾难性遗忘"问题
  训练成本太高，怎么办？
""")

# 演示：模型参数量与显存关系
print("\n[实验] 不同规模模型的训练显存")
print("-" * 60)
print(f"{'模型':<12}{'参数':<12}{'权重':<10}{'梯度+优化器':<15}{'总计':<10}")
print("-" * 60)

for params_b in [0.1, 1, 7, 70, 175]:
    weight_gb = params_b * 4
    train_state_gb = params_b * 12  # grad + Adam 状态
    total = weight_gb + train_state_gb
    print(f"{params_b}B{'':<10}{params_b:.1f}B{'':<6}{weight_gb:.0f}GB{'':<4}{train_state_gb:.0f}GB{'':<10}{total:.0f}GB")


# ============================================================
# 第2部分：LoRA 的核心思想
# ============================================================

print("\n" + "=" * 60)
print("第2部分：LoRA 的核心思想 — 低秩分解")
print("=" * 60)

print("""
LoRA 的关键观察:
  预训练模型的权重变化 ΔW 通常是"低秩"的！
  → 不需要 d×d 的更新矩阵
  → 只需要 d×r 和 r×d 的两个小矩阵

数学原理:
  原始: y = W·x            (W 是 d×d 矩阵)
  LoRA: y = W·x + (B·A)·x

  其中:
    W: 原始权重 (冻结)  - d×d
    A: 下投影 (训练)    - d×r  (r << d)
    B: 上投影 (训练)    - r×d

  参数量对比:
    原始:  d × d
    LoRA:  2 × d × r

  当 d=1024, r=8:
    原始:   1,048,576 参数
    LoRA:   16,384 参数 (少了 64x!)

类比:
  给一张照片微调:
  - 全量微调 = 重画整张照片
  - LoRA = 只画一些"微小的修改"
  - 这些修改可以用低维方式表达
""")


# ============================================================
# 第3部分：LoRA 数学详解
# ============================================================

print("\n" + "=" * 60)
print("第3部分：LoRA 数学详解")
print("=" * 60)

print("""
LoRA 的初始化:
  A: 用 Kaiming 均匀分布初始化 (类似 nn.Linear)
  B: 用 0 初始化 → 训练开始时 BA = 0
  → 训练起点 = 原始模型！保证稳定

训练目标:
  min ||ΔW - BA||²  where BA 是低秩近似

前向传播:
  output = W·x + (B·A)·x · (α/r)

  α/r 是缩放因子:
    α: LoRA 的缩放参数 (常用 16 或 32)
    r: 秩
    α/r: 保持更新幅度与 r 无关
""")

# 演示：低秩分解
print("\n[实验] 演示低秩分解的参数量")
print("-" * 60)

d = 1024  # 隐藏层维度
r = 8     # LoRA 秩

W = torch.randn(d, d)
A = torch.randn(d, r)
B = torch.randn(r, d)

orig_params = d * d
lora_params = d * r + r * d

print(f"隐藏层维度: d = {d}")
print(f"LoRA 秩:     r = {r}")
print(f"\n原始权重 W:        {d} × {d} = {orig_params:,} 参数")
print(f"LoRA 矩阵 A:       {d} × {r} = {d*r:,} 参数")
print(f"LoRA 矩阵 B:       {r} × {d} = {r*d:,} 参数")
print(f"LoRA 总参数:       {lora_params:,}")
print(f"\n参数量减少: {(1 - lora_params/orig_params)*100:.2f}%")
print(f"压缩比: {orig_params/lora_params:.0f}x")

# 验证：BA 确实可以近似一些低秩结构
print("\n[演示] 验证 BA 接近一个简单的低秩矩阵")
print("-" * 60)
# 构造一个低秩矩阵
d_demo = 32
r_demo = 4
# 目标: target[d×d] 是低秩矩阵
U = torch.randn(d_demo, r_demo)
V = torch.randn(d_demo, r_demo)
target = U @ V.T  # d×d 矩阵, 真实秩=4

# 用 LoRA 学习这个低秩矩阵
A = torch.randn(r_demo, d_demo) * 0.01  # 注意 LoRA 是 BA
B = torch.zeros(d_demo, r_demo)
BA = B @ A  # d_demo × d_demo

# 简单的梯度下降
lr = 0.001
for step in range(100):
    error = target - BA
    grad_A = -B.T @ error  # B^T: r×d, error: d×d → r×d → A
    grad_B = -error @ A.T  # error: d×d, A^T: d×r → d×r → B
    A = A - lr * grad_A
    B = B - lr * grad_B
    BA = B @ A
    if step % 20 == 0:
        print(f"步{step}: Frobenius误差 = {(target - BA).norm():.4f}")


# ============================================================
# 第4部分：LoRA 完整实现
# ============================================================

print("\n" + "=" * 60)
print("第4部分：LoRA 完整实现")
print("=" * 60)


class LoRALinear(nn.Module):
    """带 LoRA 的 Linear 层"""

    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        super().__init__()
        # 原始权重 (冻结)
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # LoRA 参数
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 保持 0 初始化 → 训练起点 = 原始模型

    def forward(self, x):
        # 原始路径 (冻结)
        result = F.linear(x, self.weight, self.bias)
        # LoRA 路径
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    def freeze_original(self):
        """冻结原始参数"""
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def unfreeze_lora(self):
        """启用 LoRA 参数训练"""
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True


# 演示 LoRA
print("\n[实验] 演示 LoRA Linear 层")
print("-" * 60)
lora_layer = LoRALinear(256, 256, r=8, alpha=16)

# 验证：初始化时, LoRA 输出为 0
x = torch.randn(2, 10, 256)
y_init = lora_layer(x)
print(f"输入 shape: {x.shape}")
print(f"输出 shape: {y_init.shape}")

# 计算参数量
total = sum(p.numel() for p in lora_layer.parameters())
trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
print(f"\n总参数: {total:,}")
print(f"可训练参数: {trainable:,} ({trainable/total*100:.2f}%)")


# ============================================================
# 第5部分：合并 LoRA 权重
# ============================================================

print("\n" + "=" * 60)
print("第5部分：合并 LoRA 权重 (部署时)")
print("=" * 60)

print("""
训练完成后，可以将 LoRA 权重合并到原始权重中：
  W_merged = W + B·A · (α/r)

合并后:
  - 模型大小不变
  - 推理速度不变
  - 不再需要 LoRA 模块
  - 可以像普通模型一样部署
""")

def merge_lora(lora_layer: LoRALinear):
    """将 LoRA 权重合并到原始权重"""
    # 计算 BA · scaling
    delta_w = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
    # 合并
    lora_layer.weight.data = lora_layer.weight.data + delta_w
    # 清空 LoRA 参数 (不需要了)
    lora_layer.lora_A.data.zero_()
    lora_layer.lora_B.data.zero_()


# 演示合并
print("\n[演示] 合并 LoRA 权重")
print("-" * 60)
lora_layer2 = LoRALinear(64, 64, r=4, alpha=8)
W_before = lora_layer2.weight.data.clone()
merge_lora(lora_layer2)
W_after = lora_layer2.weight.data.clone()
print(f"合并前 weight 范数: {W_before.norm():.4f}")
print(f"合并后 weight 范数: {W_after.norm():.4f}")
print(f"差异: {(W_after - W_before).norm():.4f} (LoRA 的贡献)")


# ============================================================
# 第6部分：在 MiniMind 中怎么用
# ============================================================

print("\n" + "=" * 60)
print("第6部分：在 MiniMind 中怎么用 LoRA")
print("=" * 60)

print("""
MiniMind 的 LoRA 微调流程:

1. 加载预训练模型
2. 冻结所有参数
3. 在 attention 层 (q_proj, v_proj) 上添加 LoRA
4. 只训练 LoRA 参数
5. 训练完成后合并权重

代码示例 (从 MiniMind 复制):
```python
from peft import LoraConfig, get_peft_model, TaskType

# 加载模型
model = AutoModelForCausalLM.from_pretrained("minimind")

# LoRA 配置
lora_config = LoraConfig(
    r=8,                    # 秩
    lora_alpha=16,          # 缩放
    target_modules=["q_proj", "v_proj"],  # 应用到哪些层
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 注入 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 1.5M || all params: 26M || trainable%: 5.77%

# 训练 (只更新 LoRA 参数)
trainer.train()

# 合并权重
model = model.merge_and_unload()
model.save_pretrained("minimind-lora")
```

可以应用 LoRA 的层:
  ✓ q_proj, k_proj, v_proj, o_proj (Attention)
  ✓ gate_proj, up_proj, down_proj (FFN)
  ✗ embedding, lm_head (通常不推荐)

rank 选择:
  r=4:  极小任务，节省更多参数
  r=8:  平衡选择 (最常用)
  r=16: 复杂任务，更强表达
  r=32+: 接近全量微调
""")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. 全量微调的成本
   - 7B 模型: 140GB 显存
   - 175B 模型: 3.5TB 显存
   - 普通 GPU 无法训练

2. LoRA 的核心思想
   - 权重变化 ΔW 是低秩的
   - 用两个小矩阵 BA 近似 ΔW
   - 参数减少 100x-1000x

3. LoRA 实现
   - 冻结原始权重 W
   - 注入可训练矩阵 A (r×d) 和 B (d×r)
   - 缩放因子 α/r
   - 训练起点 = 原始模型 (B=0)

4. 部署优化
   - 训练后合并权重
   - 模型大小不变
   - 推理速度不变

5. 在 MiniMind 中
   - 配合 peft 库使用
   - 通常只对 q_proj, v_proj 加 LoRA
   - rank=8 是常用选择

下一步: 第16课 - YaRN 长度外推
""")


# ============================================================
# 深入理解：LoRA 的"小而美"
# ============================================================
print("\n" + "=" * 60)
print("深入理解：LoRA 的'小而美'")
print("=" * 60)

print("""
【类比：LoRA = 改衣服不加布】
──────────────────────
  
  全参数微调:
    把整件衣服拆开重做
    → 改动大, 工时多, 成本高
  
  LoRA:
    在原衣服外面加一层薄薄的"内衬"
    → 看上去一样, 实际能改风格
    → 工时少, 成本低, 想换就拆
  
  数学:
    原衣服 (W) 保持不变
    内衬 (BA) 调整版型
    实际版型: W + BA

  优势:
    - 内衬可拆, 恢复原貌
    - 多套内衬应对不同场合
    - 总成本极低


【LoRA 的数学本质】
─────────────────

  低秩假设 (Intrinsic Dimension):
    模型微调其实不需要改很多方向
    虽然 W 有 d×d 个参数
    但有效的"变化方向"很少 (秩很低)

  验证:
    取 d=4096, 理论上变化有 4096 维
    实际只需要 8-64 维就够
    → 99% 的'变化方向'是冗余的

  数学表达:
    ΔW ∈ R^{d × d}, 但 rank(ΔW) ≤ r
    ΔW = B @ A, 其中 B ∈ R^{d × r}, A ∈ R^{r × d}
    参数数: d×r + r×d = 2dr, 远小于 d²


【图示：LoRA 的结构】
──────────────────

  输入 x ──┬────────── [W] ──────────┐
           │   (冻结)                  ↓
           └────────── [B] ─→ [A] ─── + → 输出
                (训练)        (训练)
                
  详细:
    A: [r, d]  先降维 (d → r)
    B: [d, r]  后升维 (r → d)
    BA: [d, d]  模拟 ΔW (但秩 ≤ r)
    
  其中:
    A 用高斯初始化
    B 初始化为 0
    → 训练开始时, LoRA 输出为 0, 模型行为不变


【LoRA 的变体】
────────────

  ┌────────────┬──────────────────┬──────────────────┐
  │ 变体        │ 描述              │ 适用              │
  ├────────────┼──────────────────┼──────────────────┤
  │ LoRA       │ 标准              │ 通用              │
  │ QLoRA      │ 4-bit + LoRA     │ 显存极紧          │
  │ DoRA       │ 分解为方向+幅度  │ 更稳定            │
  │ AdaLoRA    │ 自适应 rank      │ 高效              │
  │ LoRA+      │ A, B 不同学习率  │ 更快收敛          │
  │ rsLoRA     │ rank-stabilized  │ 训练稳定          │
  └────────────┴──────────────────┴──────────────────┘

  主流选择:
    - 单卡微调 7B 模型: QLoRA
    - 普通微调: LoRA (rank=16)
    - 实验性质: DoRA / AdaLoRA


【LoRA vs 全参数微调对比】
───────────────────────

  ┌────────────┬─────────────┬──────────────┐
  │ 指标        │ 全参数       │ LoRA         │
  ├────────────┼─────────────┼──────────────┤
  │ 显存 (7B)  │ 60+ GB      │ 16 GB        │
  │ 训练速度   │ 1x          │ 1.2x (更快!) │
  │ 参数量     │ 7B          │ 4M (0.05%)   │
  │ 存储 (每任务)│ 14 GB      │ 16 MB        │
  │ 推理速度   │ 1x          │ 1x (可合并)  │
  │ 性能       │ 100%        │ 95-99%       │
  └────────────┴─────────────┴──────────────┘

  关键洞察:
    - LoRA 显存省 70%+
    - LoRA 训练反而略快 (更新参数少)
    - 性能损失通常 < 5%
    - 推理时可合并回原模型, 无额外开销


【rank 的选择】
─────────────

  rank 越小:
    - 训练参数越少
    - 显存更省
    - 表达能力强但有限
    - 适合: 简单任务, 小数据

  rank 越大:
    - 训练参数越多
    - 显存大
    - 表达能力强
    - 适合: 复杂任务, 大数据

  经验值:
    rank=4:  极简任务 (情感分类)
    rank=8:  常见任务 (推荐)
    rank=16: 复杂任务 (对话)
    rank=32: 高度定制
    rank=64: 接近全参数

  实验 (LLaMA-7B):
    rank=8:  94% 性能
    rank=16: 97% 性能
    rank=32: 99% 性能
    rank=64: 99.5% 性能
    → 边际收益递减


【LoRA 应用场景】
──────────────

  1. 指令微调 (Instruction Tuning)
    base 模型 → 加上指令内衬 → 听话的助手
  
  2. 个性化 (Personalization)
    base 模型 → 加上不同内衬 → 不同风格
  
  3. 多任务 (Multi-task)
    1 个 base 模型 + N 套 LoRA
    → 节省 N-1 倍存储
  
  4. 持续学习 (Continual Learning)
    学新任务, 旧任务的内衬不动
    → 避免灾难性遗忘
  
  5. 部署优化
    训练时用 LoRA, 推理时合并
    → 推理速度无损失


【QLoRA：极致显存优化】
──────────────────

  4-bit 量化 base + LoRA:
    base 模型: 4-bit 存储
    LoRA 参数: FP16 训练
    → 显存省到极致

  配置 (LLaMA-65B):
    全参数: 780 GB
    LoRA:  240 GB
    QLoRA: 48 GB (单卡可跑!)

  性能:
    QLoRA: ~99% of LoRA 性能
    几乎无损失
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】LoRA 参数计算
  7B 模型, hidden=4096, rank=8
  LoRA 增加了多少参数? 占总参数多少?

【练习2】rank 选择
  任务: 100 条数据, 简单分类
  建议 rank=? 

【练习3】LoRA 目标层
  通常对哪些层加 LoRA? 为什么?

【练习4】LoRA 初始化
  为什么 A 高斯初始化, B 初始化为 0?

【练习5】QLoRA 显存
  65B 模型用 QLoRA, 大约需要多少显存?

【练习6】LoRA 合并
  训练后如何合并 LoRA 到原模型?
  合并后能否再拆开?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
d = 4096
r = 8
lora_per_layer = 2 * d * r
print(f"  hidden_size = {d}, rank = {r}")
print(f"  每层 LoRA 参数: 2 × d × r = 2 × {d} × {r} = {lora_per_layer:,}")
print()
total_7b = 7e9
print(f"  7B 模型总参数: ~{total_7b/1e9:.0f}B")
print(f"  如果对 32 层 q_proj 加 LoRA:")
total_lora = lora_per_layer * 32
ratio = total_lora / total_7b * 100
print(f"  LoRA 总参数: {lora_per_layer:,} × 32 = {total_lora:,} = {total_lora/1e6:.1f} M")
print(f"  占总参数: {ratio:.3f}%")
print()
print(f"  → 只需训练约 0.05% 的参数")
print(f"  → 节省 99.95% 的优化器状态")

# 练习2
print("\n【练习2 答案】")
print("  简单任务, 数据少 → rank 小")
print()
print("  推荐: rank=4 或 rank=8")
print()
print("  原因:")
print("    - 100 条数据能表达的'变化'有限")
print("    - rank 太大反而过拟合")
print("    - 小 rank 训练快, 不易过拟合")
print()
print("  实验:")
print("    100 条数据 + rank=64 → 过拟合")
print("    100 条数据 + rank=8  → 泛化好")
print()
print("  经验公式:")
print("    rank ≈ sqrt(数据量 / 1000)")
print("    100 条 → rank=1 (再小可以)")
print("    1000 条 → rank=1-4")
print("    10K 条 → rank=8-16")
print("    100K+ 条 → rank=16-64")

# 练习3
print("\n【练习3 答案】")
print("  常见 LoRA 目标层:")
print()
print("  LLaMA 风格:")
print("    target_modules = ['q_proj', 'v_proj']")
print("    → 最常见, 平衡效果和效率")
print()
print("  更全面:")
print("    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
print("    → 效果更好, 但参数更多")
print()
print("  全部 attention + FFN:")
print("    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',")
print("                       'gate_proj', 'up_proj', 'down_proj']")
print("    → 接近全参数效果")
print()
print("  经验:")
print("    任务简单: 只对 q_proj, v_proj")
print("    任务复杂: 加 o_proj")
print("    任务很复杂: 全部都加")
print()
print("  不加的层:")
print("    - embedding (改动会破坏词向量)")
print("    - lm_head (同 embedding)")
print("    - layernorm (参数量少, 改不改差不多)")

# 练习4
print("\n【练习4 答案】")
print("  A 高斯初始化, B 初始化为 0:")
print()
print("  训练开始时:")
print("    A ~ N(0, σ²)        (高斯, 有非零值)")
print("    B = 0                (全零)")
print("    BA = 0 * A = 0       (全零矩阵)")
print()
print("  效果:")
print("    输出 = Wx + (BA)x = Wx + 0 = Wx")
print("    → 训练开始时, LoRA 输出为 0")
print("    → 模型行为与 base 一致")
print()
print("  为什么这样?")
print("    1. 训练稳定:")
print("      不会因为 LoRA 初始化突然改变输出")
print("      loss 不会突然跳变")
print()
print("    2. 公平比较:")
print("      训练前所有 LoRA 模型行为相同")
print("      性能差异纯粹来自训练")
print()
print("    3. 优化友好:")
print("      梯度从 0 开始增大")
print("      优化器有稳定起点")
print()
print("  训练中:")
print("    B 逐渐变成非零矩阵")
print("    LoRA 开始起作用")
print("    慢慢偏离 base 模型")

# 练习5
print("\n【练习5 答案】")
print("  65B 模型用 QLoRA 的显存估算:")
print()
print("  模型参数: 65B × 0.5 字节 (4-bit) = 32.5 GB")
print("  梯度:        0 (base 冻结)")
print("  优化器:      0 (base 冻结)")
print("  LoRA 训练:   50M × 8 字节 = 0.4 GB")
print("  激活值:      ~10 GB (取决于 batch)")
print("  框架开销:    ~2 GB")
print()
total = 32.5 + 0.4 + 10 + 2
print(f"  合计: {total:.1f} GB")
print()
print("  → 单卡 A100 (80GB) 即可微调 65B 模型!")
print("  → 这就是 QLoRA 的革命性意义")
print()
print("  对比:")
print("  全参数 FP16: 65B × 2 字节 = 130 GB + 优化器等 = 780 GB")
print("  LoRA:        130 GB + LoRA")
print("  QLoRA:       48 GB (40-50 GB)")

# 练习6
print("\n【练习6 答案】")
print("  合并方法:")
print()
print("  1. peft 内置 (推荐):")
print("    merged_model = peft_model.merge_and_unload()")
print("    → 一行代码完成")
print()
print("  2. 手动合并:")
print("    delta_w = (lora_B @ lora_A) * scaling")
print("    original_weight += delta_w")
print("    lora_A.zero_()")
print("    lora_B.zero_()")
print("    → 不依赖 peft")
print()
print("  合并后能否再拆开?")
print("    理论上: 不能精确拆开")
print("    因为 W' = W + BA, 但 BA = 0 后无法恢复")
print()
print("  实际:")
print("    合并后 W' 是新的'base'")
print("    如果想换 LoRA, 需要从原始 W 开始")
print("    → 建议保留原始 base, 别扔!")
print()
print("  推理时:")
print("    合并后: 普通模型, 无 LoRA 开销")
print("    不合并: 每次推理多算一次 BA, 略慢")
print()
print("  最佳实践:")
print("    训练 → 评估 → 确认有效 → 合并 → 部署")
print("    保留 peft 版本用于继续训练")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
