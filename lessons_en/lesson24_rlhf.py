# ============================================================================
# Lesson 24: RLHF and GRPO
# ============================================================================
"""
DPO is a simplified alignment method, but RLHF is the classic paradigm.
This lesson covers the full RLHF pipeline and GRPO (Group Relative Policy Optimization).

Topics:
1. RLHF Three-Step Pipeline
2. Reward Model Training
3. PPO Policy Optimization
4. GRPO: Simplified RLHF
5. RLHF vs DPO vs GRPO Comparison

Key Concepts:
- Reward Model: Learns human preferences, scores responses
- PPO: RL algorithm that limits policy update magnitude
- GRPO: Uses group-relative rewards instead of a reward model
- KL Penalty: Prevents policy from drifting too far from reference
- Advantage Function: Measures how much better an action is than average
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. RLHF Three-Step Pipeline
# ============================================================================
"""
RLHF (Reinforcement Learning from Human Feedback) full pipeline:

Step 1: Pretrain
  → Get base language model

Step 2: SFT (Supervised Fine-Tuning)
  → Get conversational model

Step 3: RLHF Alignment
  3a. Train Reward Model
      - Collect preference data: (prompt, chosen, rejected)
      - Train model to score chosen higher than rejected

  3b. Optimize policy with PPO
      - Policy model generates responses
      - Reward model scores responses
      - PPO updates policy to generate higher-scoring responses
      - Add KL penalty to prevent drifting from reference

InstructGPT / ChatGPT uses this pipeline.
"""

print("=" * 70)
print("Lesson 24: RLHF and GRPO")
print("=" * 70)

# ============================================================================
# 2. Reward Model
# ============================================================================
"""
Reward Model goal: Learn human preferences, score responses.

Training data: (prompt, chosen_response, rejected_response)
Loss function: Bradley-Terry model

L = -log σ(r(chosen) - r(rejected))

Intuition: Make chosen reward score higher than rejected reward score.
"""


class SimpleRewardModel(nn.Module):
    """Simplified Reward Model

    Input: text → Output: scalar reward score.
    In practice, typically add a linear head on top of a language model.
    """

    def __init__(self, vocab_size=100, dim=64, n_layers=2, n_heads=4, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.reward_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        reward = self.reward_head(x[:, -1, :]).squeeze(-1)
        return reward


def train_reward_model(reward_model, data, epochs=5, lr=1e-4):
    """Train Reward Model"""
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr)
    reward_model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for sample in data:
            chosen_ids = sample['chosen_ids']
            rejected_ids = sample['rejected_ids']

            chosen_reward = reward_model(chosen_ids.unsqueeze(0))
            rejected_reward = reward_model(rejected_ids.unsqueeze(0))

            loss = -F.logsigmoid(chosen_reward - rejected_reward)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}, Reward Loss: {avg_loss:.4f}")

    return reward_model


# Train reward model
print("\n--- Reward Model Training ---")
torch.manual_seed(42)
reward_model = SimpleRewardModel()

reward_data = []
for i in range(10):
    torch.manual_seed(i * 100)
    chosen = torch.randint(3, 100, (32,))
    rejected = torch.randint(3, 100, (32,))
    reward_data.append({'chosen_ids': chosen, 'rejected_ids': rejected})

reward_model = train_reward_model(reward_model, reward_data, epochs=3, lr=1e-4)

# ============================================================================
# 3. PPO Policy Optimization
# ============================================================================
"""
PPO (Proximal Policy Optimization) core idea:

1. Policy model generates responses
2. Reward model scores responses
3. Compute advantage function
4. Use clip mechanism to limit policy update magnitude

PPO objective:
  L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

Where:
  r(θ) = π_θ(a|s) / π_old(a|s)  — new/old policy probability ratio
  A = advantage function
  ε = clip range (typically 0.2)

Plus KL penalty:
  L_total = L_ppo - β * KL(π_θ || π_ref)

Key: Clip mechanism prevents too-large policy updates, ensuring stable training.
"""


def compute_advantages(rewards, gamma=1.0):
    """Compute advantage function (simplified)

    In RLHF, usually only one response-level reward,
    so advantage = reward - baseline (mean)
    """
    baseline = rewards.mean()
    advantages = rewards - baseline
    if advantages.std() > 0:
        advantages = advantages / (advantages.std() + 1e-8)
    return advantages


def ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    """PPO Loss Function"""
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    loss = -torch.min(surr1, surr2).mean()
    return loss


def kl_penalty(log_probs_policy, log_probs_ref):
    """KL divergence penalty"""
    return (log_probs_ref - log_probs_policy).mean()


# Demo PPO loss
print("\n--- PPO Loss Calculation Example ---")
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
# 4. GRPO: Group Relative Policy Optimization
# ============================================================================
"""
GRPO (proposed by DeepSeek) is a simplified RLHF:

Core idea:
  No separate reward model needed!
  Generate multiple responses per prompt, use group-relative ranking as reward.

Pipeline:
  1. For each prompt, policy generates G responses
  2. Score each response using rules/models
  3. Normalize scores within group as advantages
  4. Update policy with PPO-style objective

Advantage calculation:
  A_i = (r_i - mean(r)) / std(r)

GRPO advantages:
  - No reward model training needed
  - Simpler training
  - Can use rule-based rewards (code execution, format checks, etc.)
"""


def grpo_advantages(rewards_per_prompt):
    """GRPO Advantage Calculation

    Args:
        rewards_per_prompt: (n_prompts, n_generations) rewards for each prompt's responses

    Returns:
        advantages: (n_prompts, n_generations) normalized advantages
    """
    mean = rewards_per_prompt.mean(dim=1, keepdim=True)
    std = rewards_per_prompt.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards_per_prompt - mean) / std
    return advantages


# Demo GRPO
print("\n--- GRPO Advantage Calculation Example ---")
rewards = torch.tensor([
    [0.8, 0.3, 0.5, 0.1],
    [0.9, 0.7, 0.2, 0.4],
    [0.6, 0.6, 0.5, 0.3],
])
advantages = grpo_advantages(rewards)
print(f"Rewards:\n{rewards}")
print(f"GRPO advantages:\n{advantages}")
print("Positive advantage = better than average, negative = worse")


def train_grpo_step(policy_model, ref_model, reward_fn, prompts,
                    n_generations=4, clip_eps=0.2, beta=0.04):
    """One step of GRPO training"""
    all_log_probs = []
    all_ref_log_probs = []
    all_rewards = []

    for prompt in prompts:
        torch.manual_seed(hash(prompt) % 10000)
        gen_log_probs = torch.randn(n_generations)
        ref_log_probs = torch.randn(n_generations)
        rewards = torch.tensor([reward_fn(prompt, i) for i in range(n_generations)])

        all_log_probs.append(gen_log_probs)
        all_ref_log_probs.append(ref_log_probs)
        all_rewards.append(rewards)

    rewards_tensor = torch.stack(all_rewards)
    advantages = grpo_advantages(rewards_tensor)

    total_loss = 0
    for i in range(len(prompts)):
        log_probs = all_log_probs[i]
        ref_probs = all_ref_log_probs[i]
        adv = advantages[i]

        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        ppo_loss_val = -torch.min(surr1, surr2).mean()

        kl = (ref_probs - log_probs).mean()
        total_loss += ppo_loss_val + beta * kl

    return total_loss / len(prompts)


def simple_reward_fn(prompt, gen_idx):
    base_score = len(prompt) / 10.0
    variation = torch.randn(1).item() * 0.3
    return base_score + variation


# Demo GRPO training
print("\n--- GRPO Training Demo ---")
prompts = ["What is AI?", "Explain quantum computing", "Python advantages"]
loss = train_grpo_step(None, None, simple_reward_fn, prompts, n_generations=4)
print(f"GRPO single-step loss: {loss.item():.4f}")

# ============================================================================
# 5. RLHF vs DPO vs GRPO Comparison
# ============================================================================
print("\n--- RLHF vs DPO vs GRPO Comparison ---")
print("""
┌──────────────┬────────────────────┬────────────────────┬────────────────────┐
│              │ RLHF (PPO)         │ DPO                │ GRPO               │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ Models needed│ 4 (policy+ref+rm+value) │ 2 (policy+ref) │ 2 (policy+ref)    │
│ Reward model │ Need to train      │ Not needed         │ Not needed         │
│ Data needed  │ Preferences+online │ Preferences only   │ Rules/model scores │
│ Training     │ On-policy (online) │ Off-policy (offline)│ On-policy (online)│
│ Stability    │ Unstable           │ Stable             │ Fairly stable      │
│ Compute cost │ High               │ Low                │ Medium             │
│ Upper bound  │ Highest            │ Good               │ Good               │
│ Notable use  │ InstructGPT        │ Llama 2            │ DeepSeek-R1        │
│ Best for     │ Large-scale align  │ Quick alignment    │ Rule-definable     │
└──────────────┴────────────────────┴────────────────────┴────────────────────┘
""")

# ============================================================================
# 6. Practical Recommendations
# ============================================================================
"""
Which alignment method to choose?

1. Quick prototype / small projects → DPO
   - Simplest, least code
   - Only needs preference data
   - Usually sufficient quality

2. Clear rules definable → GRPO
   - Math: is the answer correct?
   - Code: does it pass tests?
   - Format: does it match the format?
   - No reward model training needed

3. Large-scale production / highest quality → RLHF (PPO)
   - Theoretically best results
   - But complex training, lots of engineering
   - For well-resourced companies
"""

# ============================================================================
# Exercises
# ============================================================================
print("\n" + "=" * 70)
print("Exercises")
print("=" * 70)

# Exercise 1: Reward model loss
print("\nExercise 1: Reward model loss")
chosen_reward = torch.tensor([2.5])
rejected_reward = torch.tensor([0.8])
loss = -F.logsigmoid(chosen_reward - rejected_reward)
print(f"chosen_reward={chosen_reward.item()}, rejected_reward={rejected_reward.item()}")
print(f"Bradley-Terry loss = {loss.item():.4f}")
print(f"Verify: σ(2.5-0.8) = σ(1.7) = {torch.sigmoid(torch.tensor(1.7)).item():.4f}")

# Exercise 2: PPO clip mechanism
print("\nExercise 2: PPO clip mechanism")
ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 2.0])
advantage = torch.tensor([1.0])
eps = 0.2
clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
print(f"ratio: {ratio.tolist()}")
print(f"clipped ratio: {clipped.tolist()}")
print(f"ratio * advantage: {(ratio * advantage).tolist()}")
print(f"clipped * advantage: {(clipped * advantage).tolist()}")
print("Clip limits ratio to [0.8, 1.2], preventing too-large updates")

# Exercise 3: GRPO advantages
print("\nExercise 3: GRPO advantages")
rewards = torch.tensor([[3.0, 1.0, 2.0, 0.5]])
adv = grpo_advantages(rewards)
print(f"Rewards: {rewards.tolist()}")
print(f"GRPO advantages: {adv.tolist()}")
print(f"Highest reward gets positive advantage, lowest gets negative")

# Exercise 4: Choose alignment method
print("\nExercise 4: Choose alignment method")
print("Scenario: Training a math problem-solving model with automatic answer verification")
print("Recommendation: GRPO — math has clear correct/incorrect rules")
print("      Use 'is answer correct' as reward, no reward model needed")

print("\n" + "=" * 70)
print("Lesson 24 Summary:")
print("  1. RLHF pipeline: Pretrain → SFT → RLHF(PPO)")
print("  2. Reward model learns human preferences, scores responses")
print("  3. PPO uses clip mechanism to limit policy updates, ensuring stability")
print("  4. GRPO uses group-relative ranking instead of reward model, simpler")
print("  5. Choice: DPO (simple) / GRPO (rule-based) / RLHF (highest quality)")
print("=" * 70)
