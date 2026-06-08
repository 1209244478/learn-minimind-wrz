# ============================================================================
# 第24课：强化学习人类反馈 (RLHF) 与 GRPO
# ============================================================================
"""
DPO 是一种简化的对齐方法，但 RLHF 是更经典的对齐范式。
本课介绍 RLHF 的完整流程，以及 GRPO（Group Relative Policy Optimization）。

本课内容：
1. RLHF 的三步流程
2. 奖励模型 (Reward Model) 的训练
3. PPO 策略优化
4. GRPO：简化版 RLHF
5. RLHF vs DPO vs GRPO 对比

关键概念：
- 奖励模型 (Reward Model)：学习人类偏好，给回答打分
- PPO (Proximal Policy Optimization)：限制策略更新幅度的强化学习算法
- GRPO (Group Relative Policy Optimization)：用组内相对奖励替代奖励模型
- KL 惩罚：防止策略模型偏离参考模型太远
- 优势函数 (Advantage)：衡量某个动作比平均水平好多少
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. RLHF 的三步流程
# ============================================================================
"""
RLHF (Reinforcement Learning from Human Feedback) 的完整流程：

Step 1: 预训练 (Pretrain)
  → 得到基础语言模型

Step 2: 监督微调 (SFT)
  → 得到能对话的模型

Step 3: RLHF 对齐
  3a. 训练奖励模型 (Reward Model)
      - 收集偏好数据：(prompt, chosen, rejected)
      - 训练模型给 chosen 打高分，给 rejected 打低分

  3b. 用 PPO 优化策略模型
      - 策略模型生成回答
      - 奖励模型给回答打分
      - 用 PPO 更新策略，使其生成高分回答
      - 加 KL 惩罚防止偏离参考模型

InstructGPT / ChatGPT 就是用的这个流程。
"""

print("=" * 70)
print("第24课：强化学习人类反馈 (RLHF) 与 GRPO")
print("=" * 70)

# ============================================================================
# 2. 奖励模型 (Reward Model)
# ============================================================================
"""
奖励模型的目标：学习人类偏好，给回答打分。

训练数据：(prompt, chosen_response, rejected_response)
损失函数：Bradley-Terry 模型

L = -log σ(r(chosen) - r(rejected))

直觉：让 chosen 的奖励分数高于 rejected 的奖励分数。
"""


class SimpleRewardModel(nn.Module):
    """奖励模型

    输入一段文本，输出一个标量奖励分数。
    实际中，通常在语言模型最后一层加一个线性头。
    """

    def __init__(self, vocab_size, dim=64, n_layers=2, n_heads=4, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        # 奖励头：将隐藏状态映射为标量
        self.reward_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # 取最后一个 token 的表示作为奖励
        reward = self.reward_head(x[:, -1, :]).squeeze(-1)
        return reward


def train_reward_model(reward_model, data, epochs=5, lr=1e-4):
    """训练奖励模型

    Args:
        data: 偏好数据列表，每项包含 chosen_ids 和 rejected_ids
    """
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr)
    reward_model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for sample in data:
            chosen_ids = sample['chosen_ids']
            rejected_ids = sample['rejected_ids']

            # 获取奖励分数
            chosen_reward = reward_model(chosen_ids.unsqueeze(0))
            rejected_reward = reward_model(rejected_ids.unsqueeze(0))

            # Bradley-Terry 损失
            loss = -F.logsigmoid(chosen_reward - rejected_reward)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}, Reward Loss: {avg_loss:.4f}")

    return reward_model


# 训练奖励模型
print("\n--- 奖励模型训练 ---")

# 准备偏好数据（使用字符级 tokenizer 编码）
chosen_texts = [
    "深度学习是机器学习的子集，使用多层神经网络自动学习特征表示。",
    "Python是一种高级编程语言，以简洁易读著称。",
    "Transformer架构基于自注意力机制，能并行处理序列数据。",
]
rejected_texts = [
    "深度学习就是学得很深。",
    "Python就是蛇。",
    "Transformer就是变形金刚。",
]
# 构建字符级词表
all_chars = set()
for t in chosen_texts + rejected_texts:
    all_chars.update(t)
chars = sorted(all_chars)
char2id = {c: i + 1 for i, c in enumerate(chars)}
vocab_size = len(chars) + 1

def encode_text(text, max_len=32):
    ids = [char2id.get(c, 0) for c in text]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

reward_data = []
for chosen, rejected in zip(chosen_texts, rejected_texts):
    # 重复多次以增加训练数据
    for _ in range(3):
        reward_data.append({
            'chosen_ids': encode_text(chosen),
            'rejected_ids': encode_text(rejected),
        })

torch.manual_seed(42)
reward_model = SimpleRewardModel(vocab_size=vocab_size)
print(f"词表大小: {vocab_size}")

reward_model = train_reward_model(reward_model, reward_data, epochs=3, lr=1e-4)

# ============================================================================
# 3. PPO 策略优化
# ============================================================================
"""
PPO (Proximal Policy Optimization) 的核心思想：

1. 策略模型生成回答
2. 奖励模型给回答打分
3. 计算优势函数 (Advantage)
4. 用 clip 机制限制策略更新幅度

PPO 目标函数：
  L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

其中：
  r(θ) = π_θ(a|s) / π_old(a|s)  — 新旧策略的概率比
  A = 优势函数
  ε = clip 范围（通常 0.2）

加上 KL 惩罚：
  L_total = L_ppo - β * KL(π_θ || π_ref)

PPO 的关键：clip 机制防止策略更新太大，保证训练稳定。
"""


def compute_advantages(rewards, gamma=1.0):
    """计算优势函数（简化版）

    在 RLHF 中，通常只有一个回答级别的奖励，
    所以优势 = 奖励 - 基线（均值）

    Args:
        rewards: (batch_size,) 每个回答的奖励
        gamma: 折扣因子

    Returns:
        advantages: (batch_size,) 优势值
    """
    # 简单基线：奖励的均值
    baseline = rewards.mean()
    advantages = rewards - baseline
    # 标准化
    if advantages.std() > 0:
        advantages = advantages / (advantages.std() + 1e-8)
    return advantages


def ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    """PPO 损失函数

    Args:
        old_log_probs: 旧策略的 log 概率
        new_log_probs: 新策略的 log 概率
        advantages: 优势值
        clip_eps: clip 范围

    Returns:
        loss: PPO 损失（取负号因为要最大化）
    """
    # 计算概率比
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    # 取较小值（保守更新）
    loss = -torch.min(surr1, surr2).mean()
    return loss


def kl_penalty(log_probs_policy, log_probs_ref):
    """KL 散度惩罚

    防止策略模型偏离参考模型太远
    """
    return (log_probs_ref - log_probs_policy).mean()


# 演示 PPO 损失
print("\n--- PPO 损失计算示例 ---")
torch.manual_seed(42)
old_log_probs = torch.randn(4)
new_log_probs = torch.randn(4)
advantages = torch.tensor([1.0, -0.5, 0.8, -0.3])

loss = ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2)
print(f"old_log_probs: {old_log_probs.tolist()}")
print(f"new_log_probs: {new_log_probs.tolist()}")
print(f"advantages: {advantages.tolist()}")
print(f"PPO loss: {loss.item():.4f}")

# ============================================================================
# 4. GRPO：Group Relative Policy Optimization
# ============================================================================
"""
GRPO 是 DeepSeek 提出的简化版 RLHF：

核心思想：
  不需要单独的奖励模型！
  对同一个 prompt 生成多个回答，用组内相对排名作为奖励。

【从 PPO 到 GRPO 的简化过程】

  第1步：PPO 需要什么？
    PPO 需要一个"绝对奖励"来告诉模型"这个回答值多少分"
    所以必须先训练一个奖励模型（复杂！）

  第2步：我们真的需要绝对奖励吗？
    其实我们只关心"哪个回答更好"，不需要知道具体几分
    → 相对排名就够了！

  第3步：GRPO 的做法
    对同一个 prompt，生成 G 个回答（比如4个）
    用规则（代码执行结果、格式检查等）给每个回答打分
    在组内标准化：(分数 - 平均分) / 标准差
    → 正优势 = 比平均好，负优势 = 比平均差

  第4步：为什么不需要奖励模型？
    因为"组内相对排名"已经足够指导优化方向：
    - 比平均好的回答 → 增加概率
    - 比平均差的回答 → 降低概率
    不需要知道"好多少"，只需要知道"比平均好还是差"

  类比：
    PPO = 请专业评委给每个选手打绝对分数（需要训练评委）
    GRPO = 选手之间互相比较排名（不需要评委，只需要规则）

流程：
  1. 对每个 prompt，策略模型生成 G 个回答
  2. 用规则/模型给每个回答打分
  3. 在组内标准化分数作为优势值
  4. 用 PPO 式的目标函数更新策略

优势函数计算：
  A_i = (r_i - mean(r)) / std(r)

  逐步拆解：
  假设一个 prompt 生成了4个回答，奖励分别是 [0.8, 0.3, 0.5, 0.1]
  mean = (0.8 + 0.3 + 0.5 + 0.1) / 4 = 0.425
  std  = sqrt(((0.8-0.425)² + (0.3-0.425)² + (0.5-0.425)² + (0.1-0.425)²) / 4) ≈ 0.242

  A_1 = (0.8 - 0.425) / 0.242 ≈ +1.55  → 比平均好很多，增加概率
  A_2 = (0.3 - 0.425) / 0.242 ≈ -0.52  → 比平均差一点，降低概率
  A_3 = (0.5 - 0.425) / 0.242 ≈ +0.31  → 比平均好一点，增加概率
  A_4 = (0.1 - 0.425) / 0.242 ≈ -1.34  → 比平均差很多，降低概率

GRPO 的优点：
  - 不需要训练奖励模型
  - 训练更简单
  - 可以用规则奖励（代码执行、格式检查等）
  - 特别适合有明确评判标准的任务（数学、编程）
"""


def grpo_advantages(rewards_per_prompt):
    """GRPO 优势计算

    Args:
        rewards_per_prompt: (n_prompts, n_generations) 每个 prompt 的多个回答的奖励

    Returns:
        advantages: (n_prompts, n_generations) 标准化优势
    """
    mean = rewards_per_prompt.mean(dim=1, keepdim=True)
    std = rewards_per_prompt.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards_per_prompt - mean) / std
    return advantages


# 演示 GRPO
print("\n--- GRPO 优势计算示例 ---")
# 3 个 prompt，每个生成 4 个回答
rewards = torch.tensor([
    [0.8, 0.3, 0.5, 0.1],   # prompt 1 的 4 个回答奖励
    [0.9, 0.7, 0.2, 0.4],   # prompt 2
    [0.6, 0.6, 0.5, 0.3],   # prompt 3
])
advantages = grpo_advantages(rewards)
print(f"奖励:\n{rewards}")
print(f"GRPO 优势:\n{advantages}")
print("注意：优势在组内标准化，正优势=比平均好，负优势=比平均差")


def train_grpo_step(policy_model, ref_model, reward_fn, prompts,
                    n_generations=4, clip_eps=0.2, beta=0.04):
    """GRPO 训练的一步

    Args:
        policy_model: 策略模型
        ref_model: 参考模型（冻结）
        reward_fn: 奖励函数（可以是规则或模型）
        prompts: 一批 prompt
        n_generations: 每个 prompt 生成几个回答
        clip_eps: PPO clip 范围
        beta: KL 惩罚系数
    """
    # Step 1: 生成回答（实际中用模型生成，这里模拟）
    # 假设每个 prompt 生成 n_generations 个回答
    all_log_probs = []
    all_ref_log_probs = []
    all_rewards = []

    for prompt in prompts:
        # 模拟：为每个 prompt 生成多个回答的 log 概率
        torch.manual_seed(hash(prompt) % 10000)
        gen_log_probs = torch.randn(n_generations)
        ref_log_probs = torch.randn(n_generations)

        # 模拟：用奖励函数打分
        rewards = torch.tensor([reward_fn(prompt, i) for i in range(n_generations)])

        all_log_probs.append(gen_log_probs)
        all_ref_log_probs.append(ref_log_probs)
        all_rewards.append(rewards)

    # Step 2: 计算 GRPO 优势
    rewards_tensor = torch.stack(all_rewards)  # (n_prompts, n_generations)
    advantages = grpo_advantages(rewards_tensor)  # (n_prompts, n_generations)

    # Step 3: 计算 GRPO 损失
    total_loss = 0
    for i in range(len(prompts)):
        log_probs = all_log_probs[i]
        ref_probs = all_ref_log_probs[i]
        adv = advantages[i]

        # PPO clip 损失
        ratio = torch.exp(log_probs - log_probs.detach())  # 简化
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        ppo_loss_val = -torch.min(surr1, surr2).mean()

        # KL 惩罚
        kl = (ref_probs - log_probs).mean()

        total_loss += ppo_loss_val + beta * kl

    return total_loss / len(prompts)


# 模拟奖励函数
def simple_reward_fn(prompt, gen_idx):
    """简单的规则奖励函数"""
    # 模拟：根据 prompt 长度和生成索引给分
    base_score = len(prompt) / 10.0
    variation = torch.randn(1).item() * 0.3
    return base_score + variation


# 演示 GRPO 训练
print("\n--- GRPO 训练演示 ---")
prompts = ["什么是AI？", "解释量子计算", "Python的优点"]
loss = train_grpo_step(None, None, simple_reward_fn, prompts, n_generations=4)
print(f"GRPO 单步损失: {loss.item():.4f}")

# ============================================================================
# 5. RLHF vs DPO vs GRPO 对比
# ============================================================================
print("\n--- RLHF vs DPO vs GRPO 对比 ---")
print("""
┌──────────────┬────────────────────┬────────────────────┬────────────────────┐
│              │ RLHF (PPO)         │ DPO                │ GRPO               │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ 所需模型     │ 策略+参考+奖励+价值 │ 策略+参考           │ 策略+参考           │
│ 奖励模型     │ 需要训练            │ 不需要              │ 不需要              │
│ 数据需求     │ 偏好数据+在线生成   │ 偏好数据            │ 规则/模型评分       │
│ 训练方式     │ On-policy (在线)    │ Off-policy (离线)   │ On-policy (在线)    │
│ 训练稳定性   │ 不稳定              │ 稳定                │ 较稳定              │
│ 计算成本     │ 高                  │ 低                  │ 中                  │
│ 效果上限     │ 最高                │ 好                  │ 好                  │
│ 代表项目     │ InstructGPT         │ Llama 2            │ DeepSeek-R1        │
│ 适合场景     │ 大规模对齐          │ 快速对齐            │ 规则可定义的场景    │
└──────────────┴────────────────────┴────────────────────┴────────────────────┘
""")

# ============================================================================
# 6. 实际应用建议
# ============================================================================
"""
选择哪种对齐方法？

1. 快速原型 / 小规模项目 → DPO
   - 最简单，代码最少
   - 只需要偏好数据
   - 效果通常够用

2. 有明确规则可定义 → GRPO
   - 数学题：答案是否正确
   - 代码题：是否通过测试
   - 格式要求：是否符合格式
   - 不需要训练奖励模型

3. 大规模生产 / 最高质量 → RLHF (PPO)
   - 效果理论上最好
   - 但训练复杂，需要大量工程
   - 适合有充足资源的大公司

MiniMind 项目的选择：
  - SFT → 基础对话能力
  - DPO → 对齐人类偏好
  - GRPO → 可选，用于特定任务优化
"""

# ============================================================================
# 练习
# ============================================================================
print("\n" + "=" * 70)
print("练习")
print("=" * 70)

# 练习1：奖励模型损失
print("\n练习1：奖励模型损失计算")
chosen_reward = torch.tensor([2.5])
rejected_reward = torch.tensor([0.8])
loss = -F.logsigmoid(chosen_reward - rejected_reward)
print(f"chosen_reward={chosen_reward.item()}, rejected_reward={rejected_reward.item()}")
print(f"Bradley-Terry loss = {loss.item():.4f}")
print(f"验证：σ(2.5-0.8) = σ(1.7) = {torch.sigmoid(torch.tensor(1.7)).item():.4f}")

# 练习2：PPO clip 机制
print("\n练习2：PPO clip 机制")
ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 2.0])
advantage = torch.tensor([1.0])  # 正优势
eps = 0.2
clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
print(f"ratio: {ratio.tolist()}")
print(f"clipped ratio: {clipped.tolist()}")
print(f"ratio * advantage: {(ratio * advantage).tolist()}")
print(f"clipped * advantage: {(clipped * advantage).tolist()}")
print("clip 限制了 ratio 在 [0.8, 1.2] 范围内，防止更新过大")

# 练习3：GRPO 优势计算
print("\n练习3：GRPO 优势计算")
rewards = torch.tensor([[3.0, 1.0, 2.0, 0.5]])
adv = grpo_advantages(rewards)
print(f"奖励: {rewards.tolist()}")
print(f"GRPO 优势: {adv.tolist()}")
print(f"最高奖励的回答优势为正，最低的为负")

# 练习4：选择对齐方法
print("\n练习4：选择对齐方法")
print("场景：训练一个数学解题模型，可以自动验证答案是否正确")
print("推荐：GRPO — 因为数学题有明确的正确/错误判断规则")
print("      可以用'答案是否正确'作为奖励，无需训练奖励模型")

print("\n" + "=" * 70)
print("第24课总结：")
print("  1. RLHF 三步流程：预训练 → SFT → RLHF(PPO)")
print("  2. 奖励模型学习人类偏好，给回答打分")
print("  3. PPO 用 clip 机制限制策略更新幅度，保证训练稳定")
print("  4. GRPO 用组内相对排名替代奖励模型，更简单")
print("  5. 选择建议：DPO(简单) / GRPO(有规则) / RLHF(最高质量)")
print("=" * 70)
