# ============================================================================
# 第22课：直接偏好优化 (Direct Preference Optimization, DPO)
# ============================================================================
"""
SFT 让模型学会了回答问题，但回答的质量可能参差不齐。
DPO 的目标：让模型学会区分"好回答"和"差回答"，偏好好的回答。

本课内容：
1. 为什么需要 DPO？从 RLHF 到 DPO 的演进
2. DPO 的数学原理：Bradley-Terry 偏好模型
3. DPO 损失函数的实现
4. DPO 数据格式：chosen vs rejected
5. DPO 训练循环实现

关键概念：
- RLHF：用强化学习训练奖励模型，再优化策略模型（复杂）
- DPO：直接用偏好数据优化策略模型，无需奖励模型（简单）
- 参考模型 (Reference Model)：冻结的 SFT 模型，防止策略偏离太远
- β (beta)：控制偏好强度的超参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. 为什么需要 DPO？
# ============================================================================
"""
问题：SFT 后的模型可能生成不安全、不有帮助的回答。

传统方案：RLHF (Reinforcement Learning from Human Feedback)
  1. 训练一个奖励模型 (Reward Model) 来给回答打分
  2. 用 PPO 等强化学习算法优化策略模型
  缺点：训练流程复杂，需要4个模型（策略、参考、奖励、价值），不稳定

DPO 方案：Direct Preference Optimization
  1. 直接用"好回答 vs 差回答"的偏好数据训练
  2. 无需奖励模型，只需2个模型（策略 + 参考）
  3. 训练更稳定，代码更简单

核心直觉：
  给模型看同一个问题的两个回答，让它学会偏好好的那个。
  同时用参考模型（冻结的SFT模型）作为约束，防止模型偏离太远。
"""

print("=" * 70)
print("第22课：直接偏好优化 (DPO)")
print("=" * 70)

# ============================================================================
# 2. DPO 的数学原理
# ============================================================================
"""
DPO 的核心思想来自 Bradley-Terry 偏好模型：

人类偏好概率：
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

其中 r(x, y) 是奖励函数，y_w 是 chosen（好回答），y_l 是 rejected（差回答）

DPO 的关键推导：
  将奖励函数用策略模型和参考模型的 log 概率表示：
  r(x, y) = β * log(π(y|x) / π_ref(y|x))

  代入偏好模型，得到 DPO 损失：
  L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

直觉理解：
  - log π(y|x) - log π_ref(y|x) = 策略模型相对于参考模型的"改进程度"
  - 如果 chosen 的改进 > rejected 的改进，损失就小
  - β 控制偏好强度：β 越大，模型越倾向于偏好 chosen
"""

print("\n--- DPO 数学原理 ---")
print("DPO 损失 = -log σ(β * (log_ratio_chosen - log_ratio_rejected))")
print("其中 log_ratio = log π(y|x) - log π_ref(y|x)")
print("σ 是 sigmoid 函数")

# ============================================================================
# 3. DPO 损失函数实现
# ============================================================================


def logits_to_log_probs(logits, labels):
    """将模型输出 logits 转换为每个 token 的 log 概率

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)

    Returns:
        log_probs: (batch_size, seq_len) 每个 token 的 log 概率
    """
    log_probs = F.log_softmax(logits, dim=2)
    # 选择 labels 对应位置的 log 概率
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1):
    """DPO 损失函数

    Args:
        ref_log_probs: 参考模型的 log 概率 (batch_size, seq_len)
        policy_log_probs: 策略模型的 log 概率 (batch_size, seq_len)
        mask: 有效 token 的掩码 (batch_size, seq_len)
        beta: 偏好强度超参数

    Returns:
        loss: DPO 损失值
    """
    # 对序列维度求和（只计算有效 token）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 分开
    # 假设 batch 前半是 chosen，后半是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[:batch_size // 2]
    reject_ref = ref_log_probs[batch_size // 2:]
    chosen_policy = policy_log_probs[:batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2:]

    # 计算 log ratio
    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref

    # DPO 损失：-log σ(β * (pi_logratios - ref_logratios))
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)

    return loss.mean()


# 演示 DPO 损失计算
print("\n--- DPO 损失计算示例 ---")
torch.manual_seed(42)
B, T = 4, 8  # 2 chosen + 2 rejected

# 模拟 log 概率
ref_log_probs = torch.randn(B, T)
policy_log_probs = torch.randn(B, T)
mask = torch.ones(B, T)

loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1)
print(f"DPO 损失 (beta=0.1): {loss.item():.4f}")

# 对比不同 beta 的影响
for beta in [0.05, 0.1, 0.5, 1.0]:
    loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
    print(f"  beta={beta:.2f} -> DPO loss = {loss.item():.4f}")

# ============================================================================
# 4. DPO 数据格式
# ============================================================================
"""
DPO 数据需要成对的偏好数据：

{
  "prompt": "解释什么是量子计算",
  "chosen": [  # 好的回答
    {"role": "user", "content": "解释什么是量子计算"},
    {"role": "assistant", "content": "量子计算是利用量子力学原理..."}
  ],
  "rejected": [  # 差的回答
    {"role": "user", "content": "解释什么是量子计算"},
    {"role": "assistant", "content": "量子计算就是很快的计算机..."}
  ]
}

关键点：
- chosen 和 rejected 共享相同的 prompt
- chosen 是人类认为更好的回答
- rejected 是人类认为更差的回答
- 数据质量决定 DPO 效果
"""

print("\n--- DPO 数据格式示例 ---")
dpo_example = {
    "prompt": "解释什么是深度学习",
    "chosen_response": "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的层次化表示。它自动从原始数据中提取特征，无需人工设计特征工程。",
    "rejected_response": "深度学习就是很深的学习，学得很深。"
}
print(f"Prompt: {dpo_example['prompt']}")
print(f"Chosen (好): {dpo_example['chosen_response']}")
print(f"Rejected (差): {dpo_example['rejected_response']}")


class SimpleDPODataset:
    """DPO 数据集

    使用字符级 Tokenizer 编码 chosen 和 rejected 回答。
    """

    def __init__(self, data, max_length=32):
        self.data = data
        self.max_length = max_length
        self.pad_token_id = 0

        # 构建字符级词表
        all_chars = set()
        for sample in data:
            all_chars.update(sample["chosen_response"])
            all_chars.update(sample["rejected_response"])
        chars = sorted(all_chars)
        self.char2id = {c: i + 1 for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # +1 for pad

    def _encode_text(self, text):
        """用字符级 tokenizer 编码文本"""
        return [self.char2id.get(c, self.pad_token_id) for c in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 编码 chosen 和 rejected
        chosen_ids = self._encode_text(sample["chosen_response"])
        rejected_ids = self._encode_text(sample["rejected_response"])

        # Padding
        chosen_ids = chosen_ids[:self.max_length] + [self.pad_token_id] * max(0, self.max_length - len(chosen_ids))
        rejected_ids = rejected_ids[:self.max_length] + [self.pad_token_id] * max(0, self.max_length - len(rejected_ids))

        # 构造输入和标签（shift by 1 for next-token prediction）
        x_chosen = torch.tensor(chosen_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_ids[1:], dtype=torch.long)
        mask_chosen = (y_chosen != self.pad_token_id).long()

        x_rejected = torch.tensor(rejected_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_ids[1:], dtype=torch.long)
        mask_rejected = (y_rejected != self.pad_token_id).long()

        return {
            'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
            'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected,
        }


# ============================================================================
# 5. DPO 训练循环
# ============================================================================


class SimpleGPT(nn.Module):
    """简化 GPT 模型（用于演示 DPO 训练）

    使用课程前面讲解的组件：RMSNorm + Causal Attention + 权重共享
    """

    def __init__(self, vocab_size, dim=64, n_layers=2, n_heads=4, max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def train_dpo(policy_model, ref_model, dataset, epochs=3, lr=4e-8, beta=0.1):
    """DPO 训练循环

    关键点：
    1. ref_model 冻结，不参与梯度更新
    2. 学习率极小（4e-8），防止遗忘
    3. 每步需要4次前向传播（policy_chosen, policy_rejected, ref_chosen, ref_rejected）
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    ref_model.eval()  # 参考模型始终在 eval 模式

    policy_model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            x_chosen = batch['x_chosen']
            y_chosen = batch['y_chosen']
            mask_chosen = batch['mask_chosen']
            x_rejected = batch['x_rejected']
            y_rejected = batch['y_rejected']
            mask_rejected = batch['mask_rejected']

            # 合并 chosen 和 rejected 以提高效率
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            # 参考模型前向传播（不计算梯度）
            with torch.no_grad():
                ref_logits = ref_model(x)
                ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 策略模型前向传播
            policy_logits = policy_model(x)
            policy_log_probs = logits_to_log_probs(policy_logits, y)

            # 计算 DPO 损失
            loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}, DPO Loss: {avg_loss:.4f}")

    return policy_model


# 运行 DPO 训练
print("\n--- DPO 训练演示 ---")
dpo_data = [
    {"chosen_response": "深度学习是机器学习的子集，使用多层神经网络自动学习特征表示。",
     "rejected_response": "深度学习就是学得很深。"},
    {"chosen_response": "Python是一种高级编程语言，以简洁易读著称，广泛用于数据科学和AI。",
     "rejected_response": "Python就是蛇。"},
    {"chosen_response": "Transformer架构基于自注意力机制，能并行处理序列数据。",
     "rejected_response": "Transformer就是变形金刚。"},
]

dpo_dataset = SimpleDPODataset(dpo_data, max_length=32)
vocab_size = dpo_dataset.vocab_size

policy_model = SimpleGPT(vocab_size=vocab_size, dim=64, n_layers=2, n_heads=4, max_len=32)
# 参考模型 = 策略模型的副本（模拟 SFT 后的模型）
ref_model = SimpleGPT(vocab_size=vocab_size, dim=64, n_layers=2, n_heads=4, max_len=32)
ref_model.load_state_dict(policy_model.state_dict())  # 复制权重
ref_model.eval()
ref_model.requires_grad_(False)  # 冻结参考模型

print(f"词表大小: {vocab_size}")

print("[DPO 训练开始]")
policy_model = train_dpo(policy_model, ref_model, dpo_dataset, epochs=5, lr=1e-5, beta=0.1)

# ============================================================================
# 6. DPO vs RLHF 对比
# ============================================================================
print("\n--- DPO vs RLHF 对比 ---")
print("""
┌──────────────┬─────────────────────┬─────────────────────┐
│              │ RLHF (PPO)          │ DPO                 │
├──────────────┼─────────────────────┼─────────────────────┤
│ 所需模型     │ 4个(策略+参考+奖励+价值) │ 2个(策略+参考)      │
│ 训练复杂度   │ 高（需要on-policy采样）│ 低（off-policy）    │
│ 训练稳定性   │ 不稳定，需要调参     │ 稳定                │
│ 数据需求     │ 偏好数据 + 在线采样  │ 仅偏好数据          │
│ 计算成本     │ 高                   │ 低                  │
│ 效果上限     │ 理论上更高           │ 通常足够好          │
│ 代表项目     │ InstructGPT, ChatGPT │ Llama 2, MiniMind   │
└──────────────┴─────────────────────┴─────────────────────┘
""")

# ============================================================================
# 7. DPO 的关键超参数
# ============================================================================
"""
1. β (beta)：偏好强度
   - β 越大，模型越倾向于偏好 chosen
   - 典型值：0.1 ~ 0.5
   - β 太大：模型可能过拟合偏好数据
   - β 太小：模型几乎不学习偏好

2. 学习率：极小
   - 典型值：1e-8 ~ 5e-7
   - 比 SFT 的学习率还小1-2个数量级
   - 防止灾难性遗忘

3. 训练轮数：1-2 epochs
   - DPO 数据量通常不大
   - 过多训练会导致 chosen 的概率过高

4. 参考模型：冻结的 SFT 模型
   - 作用：防止策略模型偏离 SFT 模型太远
   - 如果没有参考模型约束，模型可能走向极端
"""

# ============================================================================
# 练习
# ============================================================================
print("\n" + "=" * 70)
print("练习")
print("=" * 70)

# 练习1：手动计算 DPO 损失
print("\n练习1：手动计算 DPO 损失")
print("给定以下 log 概率，计算 DPO 损失：")
# chosen: policy=−2.0, ref=−2.5
# rejected: policy=−3.0, ref=−2.0
chosen_policy = -2.0
chosen_ref = -2.5
reject_policy = -3.0
reject_ref = -2.0
beta = 0.1

pi_logratio = chosen_policy - reject_policy  # -2.0 - (-3.0) = 1.0
ref_logratio = chosen_ref - reject_ref  # -2.5 - (-2.0) = -0.5
logits = pi_logratio - ref_logratio  # 1.0 - (-0.5) = 1.5
loss = -math.log(1 / (1 + math.exp(-beta * logits)))
print(f"  pi_logratio = {pi_logratio}")
print(f"  ref_logratio = {ref_logratio}")
print(f"  logits = {logits}")
print(f"  DPO loss (β={beta}) = {loss:.4f}")

# 练习2：理解 β 的影响
print("\n练习2：理解 β 的影响")
print("当 β→0 时，DPO 损失趋近于什么？")
print("答案：当 β→0 时，β * logits → 0，σ(0) = 0.5，-log(0.5) = log(2) ≈ 0.693")
print("      即模型不学习任何偏好，损失恒为 log(2)")

# 练习3：DPO 数据构造
print("\n练习3：DPO 数据构造")
print("给定以下场景，构造一条 DPO 训练数据：")
print("  Prompt: '如何学习编程？'")
print("  Chosen: '建议从Python开始，它语法简洁，社区资源丰富。可以从官方教程入手...'")
print("  Rejected: '编程很难，不建议学。'")

# 练习4：log_probs 计算
print("\n练习4：log_probs 计算")
torch.manual_seed(42)
logits = torch.randn(1, 4, 10)  # (batch=1, seq_len=4, vocab=10)
labels = torch.tensor([[3, 5, 7, 2]])
log_probs = logits_to_log_probs(logits, labels)
print(f"logits shape: {logits.shape}")
print(f"labels: {labels}")
print(f"log_probs: {log_probs}")
print(f"验证：log_probs[0,0] 应该等于 log_softmax(logits)[0,0,3]")

print("\n" + "=" * 70)
print("第22课总结：")
print("  1. DPO 目标：让模型偏好好回答、远离差回答")
print("  2. DPO 损失 = -log σ(β * (log_ratio_chosen - log_ratio_rejected))")
print("  3. 参考模型（冻结的SFT模型）防止策略偏离太远")
print("  4. β 控制偏好强度，学习率极小（1e-8量级）")
print("  5. DPO 比 RLHF 更简单稳定，是当前主流对齐方法")
print("=" * 70)
