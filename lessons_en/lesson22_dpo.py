# ============================================================================
# Lesson 22: Direct Preference Optimization (DPO)
# ============================================================================
"""
SFT teaches the model to answer questions, but response quality may vary.
DPO's goal: Teach the model to distinguish "good" vs "bad" responses
and prefer the good ones.

Topics:
1. Why DPO? From RLHF to DPO
2. DPO Math: Bradley-Terry Preference Model
3. DPO Loss Function Implementation
4. DPO Data Format: chosen vs rejected
5. DPO Training Loop

Key Concepts:
- RLHF: Train a reward model with RL, then optimize policy (complex)
- DPO: Directly optimize policy with preference data, no reward model (simple)
- Reference Model: Frozen SFT model, prevents policy from drifting too far
- β (beta): Hyperparameter controlling preference strength
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. Why DPO?
# ============================================================================
"""
Problem: SFT models may generate unsafe or unhelpful responses.

Traditional: RLHF (Reinforcement Learning from Human Feedback)
  1. Train a Reward Model to score responses
  2. Use PPO to optimize the policy model
  Cons: Complex pipeline, needs 4 models (policy, ref, reward, value), unstable

DPO: Direct Preference Optimization
  1. Train directly with "good vs bad" preference pairs
  2. No reward model needed, only 2 models (policy + reference)
  3. More stable training, simpler code

Core intuition:
  Show the model two responses to the same question,
  teach it to prefer the better one.
  Use a reference model (frozen SFT) as constraint to prevent drifting.

[DPO vs RLHF: Why use DPO?]

  Imagine you're a teacher training students to write good essays:

  RLHF approach (complex):
    1. First train a "scoring teacher" (reward model) to grade essays
    2. Student writes essay -> scoring teacher grades -> student adjusts
    3. Problem: needs 4 models (policy, reference, reward, value)
    4. Training is unstable, student may learn to "game the score"

  DPO approach (simple):
    1. Just show the student two essays: "this one is good, that one is bad"
    2. Student adjusts to be more like the good one
    3. Only needs 2 models (policy + reference)
    4. Training is stable, no "gaming"

  Why can DPO replace RLHF?
    Mathematically, DPO and RLHF optimize the same objective!
    RLHF: max E[r(x,y)]  (maximize reward)
    DPO: max E[log sigma(beta * (log_ratio_chosen - log_ratio_rejected))]  (directly optimize preference)

    DPO just expresses the "reward function" implicitly via log probability ratios,
    without needing to explicitly train a reward model.
"""

print("=" * 70)
print("Lesson 22: Direct Preference Optimization (DPO)")
print("=" * 70)

# ============================================================================
# 2. DPO Math
# ============================================================================
"""
DPO is derived from the Bradley-Terry preference model:

Human preference probability:
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

where r(x, y) is the reward function, y_w is chosen, y_l is rejected.

DPO key derivation:
  Express reward using policy and reference model log probs:
  r(x, y) = β * log(π(y|x) / π_ref(y|x))

  Substituting into the preference model gives DPO loss:
  L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

Intuition:
  - log π(y|x) - log π_ref(y|x) = "improvement" of policy over reference
  - If chosen improvement > rejected improvement, loss is small
  - β controls preference strength: larger β → stronger preference
"""

print("\n--- DPO Math ---")
print("DPO loss = -log σ(β * (log_ratio_chosen - log_ratio_rejected))")
print("where log_ratio = log π(y|x) - log π_ref(y|x)")
print("σ is the sigmoid function")

# ============================================================================
# 3. DPO Loss Function Implementation
# ============================================================================


def logits_to_log_probs(logits, labels):
    """Convert model output logits to per-token log probabilities

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)

    Returns:
        log_probs: (batch_size, seq_len) log probability per token
    """
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1):
    """DPO Loss Function

    Args:
        ref_log_probs: Reference model log probs (batch_size, seq_len)
        policy_log_probs: Policy model log probs (batch_size, seq_len)
        mask: Valid token mask (batch_size, seq_len)
        beta: Preference strength hyperparameter

    Returns:
        loss: DPO loss value
    """
    # Sum over sequence dimension (only valid tokens)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # Split chosen and rejected (first half chosen, second half rejected)
    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[:batch_size // 2]
    reject_ref = ref_log_probs[batch_size // 2:]
    chosen_policy = policy_log_probs[:batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2:]

    # Compute log ratios
    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref

    # DPO loss: -log σ(β * (pi_logratios - ref_logratios))
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)

    return loss.mean()


# Demo DPO loss
print("\n--- DPO Loss Calculation Example ---")
torch.manual_seed(42)
B, T = 4, 8  # 2 chosen + 2 rejected

ref_log_probs = torch.randn(B, T)
policy_log_probs = torch.randn(B, T)
mask = torch.ones(B, T)

loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1)
print(f"DPO loss (beta=0.1): {loss.item():.4f}")

for beta in [0.05, 0.1, 0.5, 1.0]:
    loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
    print(f"  beta={beta:.2f} -> DPO loss = {loss.item():.4f}")

# ============================================================================
# 4. DPO Data Format
# ============================================================================
"""
DPO data requires paired preference data:

{
  "prompt": "Explain quantum computing",
  "chosen": [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."}
  ],
  "rejected": [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing is just fast computers..."}
  ]
}
"""

print("\n--- DPO Data Format Example ---")
dpo_example = {
    "prompt": "Explain deep learning",
    "chosen_response": "Deep learning is a subset of ML using multi-layer neural networks to learn hierarchical representations from data automatically.",
    "rejected_response": "Deep learning is just learning very deeply."
}
print(f"Prompt: {dpo_example['prompt']}")
print(f"Chosen (good): {dpo_example['chosen_response']}")
print(f"Rejected (bad): {dpo_example['rejected_response']}")


class SimpleDPODataset:
    """Simplified DPO Dataset"""

    def __init__(self, data, vocab_size=100, max_length=32):
        self.data = data
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0

    def __len__(self):
        return len(self.data)

    def _encode_response(self, text, seed):
        torch.manual_seed(seed)
        length = min(len(text) + 5, self.max_length)
        ids = torch.randint(3, self.vocab_size, (length,)).tolist()
        ids = ids + [self.pad_token_id] * (self.max_length - len(ids))
        return ids[:self.max_length]

    def __getitem__(self, idx):
        sample = self.data[idx]
        chosen_ids = self._encode_response(sample["chosen_response"], seed=idx * 2)
        rejected_ids = self._encode_response(sample["rejected_response"], seed=idx * 2 + 1)

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
# 5. DPO Training Loop
# ============================================================================


class SimpleGPT(nn.Module):
    """Simplified GPT model (for DPO training demo)"""

    def __init__(self, vocab_size=100, dim=64, n_layers=2, n_heads=4, max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
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
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def train_dpo(policy_model, ref_model, dataset, epochs=3, lr=4e-8, beta=0.1):
    """DPO Training Loop

    Key points:
    1. ref_model is frozen, no gradient updates
    2. Very small learning rate (4e-8), prevents forgetting
    3. Each step needs 4 forward passes (policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    ref_model.eval()

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

            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            with torch.no_grad():
                ref_logits = ref_model(x)
                ref_log_probs = logits_to_log_probs(ref_logits, y)

            policy_logits = policy_model(x)
            policy_log_probs = logits_to_log_probs(policy_logits, y)

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


# Run DPO training
print("\n--- DPO Training Demo ---")
dpo_data = [
    {"chosen_response": "Deep learning is a subset of ML using multi-layer neural networks to automatically learn feature representations.",
     "rejected_response": "Deep learning is just learning very deeply."},
    {"chosen_response": "Python is a high-level programming language known for its readability, widely used in data science and AI.",
     "rejected_response": "Python is just a snake."},
    {"chosen_response": "Transformer architecture is based on self-attention, enabling parallel processing of sequences.",
     "rejected_response": "Transformer is just a robot movie."},
]

policy_model = SimpleGPT(vocab_size=100, dim=64, n_layers=2, n_heads=4, max_len=32)
ref_model = SimpleGPT(vocab_size=100, dim=64, n_layers=2, n_heads=4, max_len=32)
ref_model.load_state_dict(policy_model.state_dict())
ref_model.eval()
ref_model.requires_grad_(False)

dpo_dataset = SimpleDPODataset(dpo_data, vocab_size=100, max_length=32)

print("[DPO Training Start]")
policy_model = train_dpo(policy_model, ref_model, dpo_dataset, epochs=5, lr=1e-5, beta=0.1)

# ============================================================================
# 6. DPO vs RLHF Comparison
# ============================================================================
print("\n--- DPO vs RLHF Comparison ---")
print("""
┌──────────────┬─────────────────────┬─────────────────────┐
│              │ RLHF (PPO)          │ DPO                 │
├──────────────┼─────────────────────┼─────────────────────┤
│ Models needed│ 4 (policy+ref+rm+value) │ 2 (policy+ref)  │
│ Complexity   │ High (on-policy)    │ Low (off-policy)    │
│ Stability    │ Unstable            │ Stable              │
│ Data needed  │ Preferences + online│ Preferences only    │
│ Compute cost │ High                │ Low                 │
│ Upper bound  │ Theoretically higher│ Usually sufficient  │
│ Notable use  │ InstructGPT/ChatGPT │ Llama 2, MiniMind   │
└──────────────┴─────────────────────┴─────────────────────┘
""")

# ============================================================================
# Exercises
# ============================================================================
print("\n" + "=" * 70)
print("Exercises")
print("=" * 70)

# Exercise 1: Manual DPO loss
print("\nExercise 1: Manual DPO loss calculation")
chosen_policy = -2.0
chosen_ref = -2.5
reject_policy = -3.0
reject_ref = -2.0
beta = 0.1

pi_logratio = chosen_policy - reject_policy
ref_logratio = chosen_ref - reject_ref
logits = pi_logratio - ref_logratio
loss = -math.log(1 / (1 + math.exp(-beta * logits)))
print(f"  pi_logratio = {pi_logratio}")
print(f"  ref_logratio = {ref_logratio}")
print(f"  logits = {logits}")
print(f"  DPO loss (β={beta}) = {loss:.4f}")

# Exercise 2: β impact
print("\nExercise 2: Impact of β")
print("When β→0, what does DPO loss approach?")
print("Answer: When β→0, β * logits → 0, σ(0) = 0.5, -log(0.5) = log(2) ≈ 0.693")
print("      The model learns no preference, loss is constant at log(2)")

# Exercise 3: DPO data construction
print("\nExercise 3: DPO data construction")
print("Scenario: Training a coding assistant")
print("  Prompt: 'How to sort a list in Python?'")
print("  Chosen: 'Use sorted() for a new list or .sort() to sort in place.'")
print("  Rejected: 'Just use a for loop to compare elements.'")

# Exercise 4: log_probs computation
print("\nExercise 4: log_probs computation")
torch.manual_seed(42)
logits = torch.randn(1, 4, 10)
labels = torch.tensor([[3, 5, 7, 2]])
log_probs = logits_to_log_probs(logits, labels)
print(f"logits shape: {logits.shape}")
print(f"labels: {labels}")
print(f"log_probs: {log_probs}")

print("\n" + "=" * 70)
print("Lesson 22 Summary:")
print("  1. DPO goal: Make model prefer good responses, avoid bad ones")
print("  2. DPO loss = -log σ(β * (log_ratio_chosen - log_ratio_rejected))")
print("  3. Reference model (frozen SFT) prevents policy from drifting")
print("  4. β controls preference strength, LR is very small (~1e-8)")
print("  5. DPO is simpler and more stable than RLHF, now the mainstream method")
print("=" * 70)
