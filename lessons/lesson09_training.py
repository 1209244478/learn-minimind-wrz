"""
第9课：训练循环 — 让模型学会说话
==================================

模型有了，数据有了，怎么让模型"学会"？
训练循环就是：反复给模型看数据，让它越来越准。

核心流程：
  for epoch in range(num_epochs):
      for batch in dataloader:
          1. 前向传播: 用当前参数计算预测
          2. 计算损失: 预测和真实标签的差距
          3. 反向传播: 计算梯度（每个参数该往哪调）
          4. 更新参数: 沿梯度方向微调参数

关键概念：
  - 学习率: 每步参数调整的幅度，太大不稳定，太小学得慢
  - 梯度裁剪: 防止梯度过大导致训练崩溃
  - 学习率调度: 训练过程中动态调整学习率

运行: python lessons/lesson09_training.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 复用第8课的模型（简化版）
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def precompute_freqs_cis(dim, end=2048, rope_base=1e6):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    cos_h, sin_h = cos[..., :half], sin[..., :half]
    q_out = torch.cat([q1 * cos_h - q2 * sin_h, q2 * cos_h + q1 * sin_h], dim=-1)
    k_out = torch.cat([k1 * cos_h - k2 * sin_h, k2 * cos_h + k1 * sin_h], dim=-1)
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if self.n_rep > 1:
            xk = xk[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, seq_len, self.num_heads, self.head_dim)
            xv = xv[:, :, :, None, :].expand(bsz, seq_len, self.num_kv_heads, self.n_rep, self.head_dim).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores += mask
        weights = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq))
        output = weights @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        residual = x
        x = residual + self.self_attn(self.input_layernorm(x), cos, sin)
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMindGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 num_kv_heads, head_dim, intermediate_size, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        cos = self.freqs_cos[:seq_len]
        sin = self.freqs_sin[:seq_len]
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


# ============================================================
# 第一步：准备训练数据
# ============================================================

print("=" * 60)
print("实验1：准备训练数据")
print("=" * 60)

vocab_size = 200
seq_len = 32

# 生成一个简单的模式数据：重复的数字序列
# 模型应该学会这个模式
torch.manual_seed(42)
pattern = torch.randint(0, vocab_size, (seq_len,))
train_data = pattern.repeat(100)

print(f"训练数据长度: {len(train_data)} tokens")
print(f"序列长度: {seq_len}")
print(f"词表大小: {vocab_size}")
print(f"模式: {pattern[:10].tolist()}... (重复100次)")


# ============================================================
# 第二步：最简单的训练循环
# ============================================================

print("\n" + "=" * 60)
print("实验2：最简单的训练循环")
print("=" * 60)

model = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"优化器: AdamW, 学习率: 1e-3")
print(f"初始期望损失: ln({vocab_size}) = {math.log(vocab_size):.4f}")
print()

# 训练 50 步
losses = []
for step in range(50):
    # 随机取一个 batch
    start = torch.randint(0, len(train_data) - seq_len - 1, (1,)).item()
    batch = train_data[start:start + seq_len + 1].unsqueeze(0)

    input_ids = batch[:, :-1]
    targets = batch[:, 1:]

    # 前向传播
    logits = model(input_ids)

    # 计算损失
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新参数
    optimizer.step()

    losses.append(loss.item())
    if step % 10 == 0 or step == 49:
        print(f"Step {step:3d}: loss = {loss.item():.4f}")

print(f"\n损失从 {losses[0]:.4f} 降到 {losses[-1]:.4f}")
print("→ 模型在学习！损失在下降！")


# ============================================================
# 第三步：梯度裁剪 — 防止训练崩溃
# ============================================================

print("\n" + "=" * 60)
print("实验3：梯度裁剪")
print("=" * 60)

# 梯度裁剪：限制梯度的最大范数
# 防止某些步梯度过大，导致参数更新太猛

model2 = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

# 不裁剪 vs 裁剪
input_ids = torch.randint(0, vocab_size, (2, seq_len))
targets = torch.randint(0, vocab_size, (2, seq_len))

logits = model2(input_ids)
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
loss.backward()

# 查看梯度范数
total_norm = 0
for p in model2.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5

print(f"裁剪前梯度总范数: {total_norm:.4f}")

# 梯度裁剪
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm)

total_norm_after = 0
for p in model2.parameters():
    if p.grad is not None:
        total_norm_after += p.grad.data.norm(2).item() ** 2
total_norm_after = total_norm_after ** 0.5

print(f"裁剪后梯度总范数: {total_norm_after:.4f} (max_norm={max_norm})")
print(f"\n梯度裁剪确保每步更新不会太猛，防止训练崩溃")


# ============================================================
# 第四步：学习率调度
# ============================================================

print("\n" + "=" * 60)
print("实验4：学习率调度 — Cosine Annealing")
print("=" * 60)

# MiniMind 使用余弦退火调度：
# 学习率先线性增加（warmup），再按余弦曲线衰减

class CosineScheduler:
    """余弦退火学习率调度器"""

    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, step):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


model3 = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
optimizer3 = torch.optim.AdamW(model3.parameters(), lr=0)
scheduler = CosineScheduler(optimizer3, warmup_steps=10, max_steps=100,
                            max_lr=1e-3, min_lr=1e-5)

print("学习率变化:")
print(f"{'步数':>6} | {'学习率':>10}")
print("-" * 20)
for step in [0, 5, 10, 25, 50, 75, 100]:
    lr = scheduler.get_lr(step)
    print(f"  {step:>4} | {lr:>10.6f}")

print("\nWarmup阶段 (0-10步): 学习率从0线性增到最大值")
print("衰减阶段 (10-100步): 学习率按余弦曲线从最大值降到最小值")


# ============================================================
# 第五步：完整的训练循环
# ============================================================

print("\n" + "=" * 60)
print("实验5：完整的训练循环（含所有技巧）")
print("=" * 60)

model4 = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
optimizer4 = torch.optim.AdamW(model4.parameters(), lr=0, weight_decay=0.01)
scheduler4 = CosineScheduler(optimizer4, warmup_steps=5, max_steps=100,
                             max_lr=5e-4, min_lr=1e-5)

num_steps = 100
batch_size = 4
max_grad_norm = 1.0

print(f"配置:")
print(f"  优化器: AdamW (weight_decay=0.01)")
print(f"  学习率: 5e-4 → 1e-5 (余弦退火, warmup=5步)")
print(f"  梯度裁剪: max_norm={max_grad_norm}")
print(f"  总步数: {num_steps}")
print()

losses_full = []
for step in range(num_steps):
    # 学习率调度
    lr = scheduler4.step(step)

    # 取 batch
    starts = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,))
    batch = torch.stack([train_data[s:s + seq_len + 1] for s in starts])
    input_ids = batch[:, :-1]
    targets = batch[:, 1:]

    # 前向传播
    logits = model4(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    # 反向传播
    optimizer4.zero_grad()
    loss.backward()

    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(model4.parameters(), max_grad_norm)

    # 更新参数
    optimizer4.step()

    losses_full.append(loss.item())
    if step % 20 == 0 or step == 99:
        print(f"Step {step:3d}: loss={loss.item():.4f}, lr={lr:.6f}, grad_norm={grad_norm:.4f}")

print(f"\n损失: {losses_full[0]:.4f} → {losses_full[-1]:.4f}")
print("训练让模型学会了数据中的模式！")


# ============================================================
# 第六步：验证训练效果
# ============================================================

print("\n" + "=" * 60)
print("实验6：验证训练效果")
print("=" * 60)

# 用训练好的模型预测
model4.eval()
with torch.no_grad():
    test_input = pattern[:8].unsqueeze(0)
    logits = model4(test_input)
    predictions = logits.argmax(dim=-1)

print(f"输入:    {test_input[0].tolist()}")
print(f"预测:    {predictions[0].tolist()}")
print(f"真实下一词: {pattern[1:9].tolist()}")

correct = (predictions[0] == pattern[1:9]).sum().item()
print(f"正确: {correct}/8")

# 计算困惑度 (Perplexity)
with torch.no_grad():
    full_input = pattern.unsqueeze(0)
    logits = model4(full_input)
    shift_logits = logits[:, :-1, :]
    shift_labels = pattern[1:].unsqueeze(0)
    loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))
    perplexity = math.exp(loss.item())

print(f"\n困惑度 (Perplexity): {perplexity:.2f}")
print(f"随机模型困惑度: {vocab_size} (= 词表大小)")
print(f"困惑度越低，模型预测越准确")


# ============================================================
# 第七步：训练中的关键概念总结
# ============================================================

print("\n" + "=" * 60)
print("实验7：训练关键概念")
print("=" * 60)

print("""
1. 前向传播: 输入 → 模型 → logits → loss
2. 反向传播: loss → 计算每个参数的梯度
3. 参数更新: 参数 = 参数 - lr × 梯度
4. 梯度裁剪: 防止梯度过大导致训练崩溃
5. 学习率调度: warmup + 余弦衰减
6. 权重衰减 (weight_decay): 防止参数过大，提升泛化

MiniMind 的训练配置:
  优化器: AdamW (weight_decay=0.01)
  学习率: 5e-4 (余弦退火)
  Batch size: 64
  序列长度: 512
  梯度裁剪: max_norm=1.0
  训练步数: ~50K steps
""")


# ============================================================
# 深入理解：训练循环的本质
# ============================================================
print("\n" + "=" * 60)
print("深入理解：训练循环的本质")
print("=" * 60)

print("""
【类比：训练就像"教学生"】
─────────────────────
  模型 = 学生 (大脑空白)
  训练数据 = 教材
  Loss = 考试分数 (越低越好)
  梯度 = 错题分析 (告诉学生哪里错了)
  优化器 = 学习方法 (怎么改正)

  流程:
    1. 老师讲课 (前向) → 学生答题
    2. 批改试卷 (loss)
    3. 分析错题 (反向)
    4. 学生改进 (参数更新)
    5. 重复, 直至掌握


【图示：训练循环】
────────────────

  ┌────────────────────────────────────┐
  │         训练循环 (重复 N 次)         │
  │                                    │
  │  ┌──────────┐  ┌──────────┐         │
  │  │ Forward  │→ │Loss 计算 │         │
  │  └─────┬────┘  └────┬─────┘         │
  │        │            │               │
  │        ▼            │               │
  │  ┌──────────┐       │               │
  │  │ Backward │←──────┘               │
  │  │ (梯度)   │                       │
  │  └─────┬────┘                       │
  │        │                            │
  │        ▼                            │
  │  ┌──────────┐                       │
  │  │ Optimizer│ (AdamW 等)            │
  │  │ 更新参数 │                       │
  │  └─────┬────┘                       │
  │        │                            │
  │        └──── 回到 Forward ──────    │
  └────────────────────────────────────┘


【Loss 不下降的常见原因】
─────────────────────
  1. 学习率太大:
     现象: Loss 抖动, 甚至发散
     解决: 减小 lr (如 1e-4 → 1e-5)
  
  2. 学习率太小:
     现象: Loss 几乎不动
     解决: 增大 lr, 或用更大学习率 warmup
  
  3. 数据问题:
     现象: 训练集 loss 也不下降
     解决: 检查数据, 看 loss 初始值是否正确
  
  4. 模型太小:
     现象: Loss 下降但停在高位
     解决: 用更大的模型
  
  5. 训练不够:
     现象: 还在下降趋势中
     解决: 继续训练
  
  6. Batch 太大:
     现象: Loss 下降但泛化差
     解决: 减小 batch size


【学习率调度的直观理解】
────────────────────
  类比: 学骑车
  
  Warmup 阶段 (开始时):
    学习率从 0 慢慢增大
    原因: 模型刚开始, 不能步子太大
    类比: 刚学骑车, 慢一点找平衡
  
  稳定阶段 (中期):
    学习率保持较大
    类比: 学会基本动作, 大胆尝试
  
  Decay 阶段 (后期):
    学习率逐渐减小
    类比: 已经熟练, 精修细节

  曲线 (理想):
  
  lr│   /\
     │  /  \___________
     │ /              \_______
     │/                       \____
     └──────────────────────────────── step
       warmup  cosine decay


【梯度累积 (Gradient Accumulation)】
──────────────────────────────────
  问题: 显存不够, 装不下大 batch
  解决: 模拟大 batch
  
  真实 batch = 4
  累积步数 = 4
  等效 batch = 16
  
  步骤:
    step 1: forward + backward (累积梯度)
    step 2: forward + backward (累积梯度)
    step 3: forward + backward (累积梯度)
    step 4: forward + backward (累积梯度)
    step 5: optimizer.step() (用累积的梯度更新)

  → 用小显存模拟大 batch, 训练效果等价
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】Loss 不下降
  训练时 loss 一直是 NaN, 可能的原因?

【练习2】梯度裁剪
  为什么需要梯度裁剪? 设 max_norm=1.0 意味着什么?

【练习3】学习率与 batch size
  增大 batch size 4 倍, 学习率应该怎么调整?

【练习4】梯度累积
  显存只能 batch=2, 想要等效 batch=32, 需要累积几步?

【练习5】过拟合与欠拟合
  训练集 loss 持续下降, 验证集 loss 先降后升
  这是什么现象? 该怎么解决?

【练习6】AdamW vs SGD
  简单比较 AdamW 和 SGD 在 LLM 训练中的优缺点
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  Loss = NaN 的可能原因:")
print()
print("  1. 学习率过大:")
print("    - 梯度爆炸, 参数变成 inf/nan")
print("    - 解决: 减小 lr 10x")
print()
print("  2. 数据有 NaN:")
print("    - 某些 token ID 超出 vocab_size")
print("    - 数值计算出现 0/0")
print("    - 解决: 清洗数据")
print()
print("  3. 数值溢出:")
print("    - 注意力分数太大, softmax 后是 NaN")
print("    - 解决: 加梯度裁剪, 或缩放 loss")
print()
print("  4. 模型初始化不当:")
print("    - 某些权重初始为 0 或极端值")
print("    - 解决: 检查初始化")
print()
print("  调试技巧:")
print("    - 先用 lr=1e-6 试, 看 loss 是否稳定")
print("    - 单步调试, 打印中间值")
print("    - 监控梯度范数")

# 练习2
print("\n【练习2 答案】")
print("  梯度裁剪的原因:")
print()
print("  1. 防止梯度爆炸:")
print("    - 深层网络, 梯度可能指数级增长")
print("    - 一次大的梯度更新会让模型'跳到'坏区域")
print("    - Loss 突然飙升, 难以恢复")
print()
print("  2. max_norm=1.0 含义:")
print("    - 如果所有参数梯度的 L2 范数 > 1.0")
print("    - 把梯度按比例缩小到范数 = 1.0")
print("    - 不改变方向, 只缩小幅度")
print()
print("  公式: grad = grad * min(1, max_norm / ||grad||)")
print()
print("  实验对比:")
print("    无裁剪: loss 偶尔 spike, 不稳定")
print("    有裁剪: loss 平滑下降, 训练稳定")

# 演示
import torch
import torch.nn as nn
torch.manual_seed(42)
m = nn.Linear(4, 4)
x = torch.randn(1, 4)
y = torch.randn(1, 4)
loss = ((m(x) - y) ** 2).sum()
loss.backward()
grad_norm = sum(p.grad.norm() ** 2 for p in m.parameters()) ** 0.5
print(f"  示例梯度范数: {grad_norm:.4f}")
print(f"  裁剪到 max_norm=1.0 后范数: 1.0000")

# 练习3
print("\n【练习3 答案】")
print("  经验法则: 线性缩放 (linear scaling rule)")
print()
print("  原来:  bs=32,  lr=1e-4")
print("  现在:  bs=128, lr=4e-4  (4倍)")
print()
print("  原因:")
print("    - 大 batch 的梯度是更准确的'平均'")
print("    - 不需要那么保守的学习率")
print("    - 但太大也会不稳定")
print()
print("  实际中常用 sqrt 缩放:")
print("    bs 增 4 倍, lr 增 2 倍")
print("    更保守, 更稳定")
print()
print("  还需要:")
print("    - warmup 步数按比例增加")
print("    - 因为大 batch 等效更'快'")

# 练习4
print("\n【练习4 答案】")
real_batch = 2
target_batch = 32
accum_steps = target_batch // real_batch
print(f"  实际 batch = {real_batch}")
print(f"  目标 batch = {target_batch}")
print(f"  累积步数 = {target_batch} / {real_batch} = {accum_steps} 步")
print()
print("  代码流程:")
print("    for step in range(accum_steps):")
print("        loss = model(batch)")
print("        loss = loss / accum_steps  # 缩放")
print("        loss.backward()           # 累积梯度")
print("    optimizer.step()             # 更新参数")
print("    optimizer.zero_grad()")
print()
print("  注意事项:")
print("    - loss 要除以 accum_steps, 等效大 batch")
print("    - 最后一个 step 后再 optimizer.step()")
print("    - BatchNorm 用大 batch 更准 (LLM 用 RMSNorm 无此问题)")

# 练习5
print("\n【练习5 答案】")
print("  现象: 过拟合 (Overfitting)")
print()
print("  表现:")
print("    训练集 loss: ▁▁▁▁▁ (持续下降)")
print("    验证集 loss: ▁▁╱╲___ (先降后升)")
print("                       ↑")
print("                  这里开始过拟合")
print()
print("  原因:")
print("    - 模型学到了训练集的'噪声'")
print("    - 训练集太特殊, 验证集没见过")
print("    - 模型泛化能力差")
print()
print("  解决方法:")
print("    1. 增加数据 (最有效)")
print("    2. 减小模型 (降低容量)")
print("    3. 正则化:")
print("       - Dropout (训练时随机丢弃)")
print("       - Weight Decay (惩罚大参数)")
print("       - 数据增强")
print("    4. Early Stopping:")
print("       - 验证集 loss 上升时停止训练")
print("    5. 简化任务 (如果可能)")
print()
print("  LLM 的情况:")
print("    - 数据量极大, 很少过拟合")
print("    - 主要担心是'欠拟合' (训练不够)")
print("    - 但小模型在小数据上仍会过拟合")

# 练习6
print("\n【练习6 答案】")
print("  SGD (随机梯度下降):")
print("    优点:")
print("      - 简单, 显存占用少")
print("      - 训练稳定")
print("      - 泛化性能好")
print("    缺点:")
print("      - 收敛慢")
print("      - 在鞍点处可能停滞")
print("      - 对学习率敏感")
print()
print("  AdamW (Adam + Weight Decay):")
print("    优点:")
print("      - 自适应学习率 (每个参数不同 lr)")
print("      - 收敛快")
print("      - 对超参不太敏感")
print("    缺点:")
print("      - 显存占用大 (存一阶二阶动量)")
print("      - 泛化性能略差于 SGD")
print("      - 偶尔训练不稳定")
print()
print("  实际选择:")
print("    LLM 预训练: AdamW (主流, 收敛快)")
print("    视觉微调: SGD 或 AdamW")
print("    小数据集: AdamW 容易过拟合, 用 SGD 更稳")
print()
print("  进阶:")
print("    - LION (2023): 显存更小, 速度更快")
print("    - Muon (2024): 用 Newton-Schulz 正交化, 效果更好")
print("    - 这些在 MiniMind 第12课会讲到")


# ============================================================
# 本课小结
# ============================================================
print("=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. 训练循环 = 前向传播 + 计算损失 + 反向传播 + 更新参数
2. 损失下降 = 模型在学习
3. 梯度裁剪防止训练崩溃
4. 学习率调度：warmup + 余弦衰减
5. 困惑度衡量模型质量，越低越好

完整训练流程：
  数据 → 模型 → logits → loss → 梯度 → 更新参数 → 重复
  "你好" → 模型 → 预测"世" → 错了 → 调整 → 下次更准

下一步 → lesson10_generation.py：让模型一个字一个字地写
""")
