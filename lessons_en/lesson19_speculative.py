"""
Lesson 19: Speculative Decoding — Draft-then-Verify for Faster Generation
==========================================================================

Problem: LLM autoregressive generation is slow
  - Each token requires a full forward pass
  - 7B model generating 100 tokens ≈ 100 forward passes
  - GPU utilization is very low (memory-bound, not compute-bound)

Speculative Decoding solution:
  - Use a small "draft model" to quickly generate K candidate tokens
  - Use the large "target model" to verify all K tokens in one pass
  - Accept matching tokens, reject and resample mismatched ones
  - Theoretical guarantee: Output distribution is identical to the target model

Speedup: 2-3x (depends on draft model accuracy)

This lesson covers:
  1. Why autoregressive generation is slow
  2. Speculative decoding algorithm
  3. Acceptance/rejection mechanism
  4. Complete implementation

Run: python lessons_en/lesson19_speculative.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# Part 1: Why Autoregressive Generation is Slow
# ============================================================

print("=" * 60)
print("Part 1: Why Autoregressive Generation is Slow")
print("=" * 60)

print("""
Autoregressive generation:
  for i in range(max_new_tokens):
      logits = model(input_ids)     # Full forward pass
      next_token = sample(logits[-1])
      input_ids.append(next_token)

Problem:
  - Each token requires a full forward pass
  - 100 tokens = 100 forward passes
  - GPU is memory-bound: Most time is spent reading weights
  - Compute units are idle most of the time

Analogy:
  Autoregressive = Asking a professor one question at a time
    -> Each time: Walk to office, wait, ask, walk back
    -> Very inefficient

  Speculative = Preparing multiple questions, verifying all at once
    -> Draft model quickly writes K answers
    -> Professor verifies all K at once
    -> Much more efficient
""")

print("\n[Demo] GPU utilization during generation")
print("-" * 60)
print(f"{'Method':<25}{'Forward Passes':<15}{'GPU Utilization':<15}")
print("-" * 60)
print(f"{'Autoregressive (100 tok)':<25}{'100':<15}{'~10%':<15}")
print(f"{'Speculative (K=5)':<25}{'~25':<15}{'~40%':<15}")
print(f"{'Batch inference (B=8)':<25}{'100':<15}{'~80%':<15}")


# ============================================================
# Part 2: Speculative Decoding Algorithm
# ============================================================

print("\n" + "=" * 60)
print("Part 2: Speculative Decoding Algorithm")
print("=" * 60)

print("""
Speculative Decoding flow:

  1. Draft model generates K tokens (fast, small model)
     x_1, x_2, ..., x_K = draft_model.generate(input, K)

  2. Target model verifies K tokens in one forward pass
     p_1, p_2, ..., p_K, p_{K+1} = target_model(input + x_1...x_K)

  3. Accept/reject each token:
     For i = 1, 2, ..., K:
       if random() < min(1, p_target(x_i) / p_draft(x_i)):
         Accept x_i
       else:
         Reject x_i, resample from adjusted distribution
         Break (no further checking)

  4. If all K accepted, also take p_{K+1} (bonus token)

Key insight:
  - Target model's forward pass processes K+1 positions simultaneously
  - This costs roughly the same as 1 autoregressive step
  - If draft accuracy is 80%, average acceptance ≈ 4/5 tokens
  - Effective speedup: K * acceptance_rate / 1 ≈ 2-3x

Mathematical guarantee:
  The output distribution is EXACTLY the same as the target model
  -> No quality loss, only speed gain
""")


# ============================================================
# Part 3: Acceptance/Rejection Mechanism
# ============================================================

print("\n" + "=" * 60)
print("Part 3: Acceptance/Rejection Mechanism")
print("=" * 60)

print("""
Acceptance criterion:
  Accept if: random() < min(1, p_target(x) / p_draft(x))

  Intuition:
    - If target and draft agree (p_target ≈ p_draft): ratio ≈ 1, always accept
    - If target likes x more than draft (p_target > p_draft): ratio > 1, always accept
    - If target dislikes x (p_target < p_draft): ratio < 1, may reject

Rejection resampling:
  When x_i is rejected, sample from the adjusted distribution:
    p_adjusted(x) = max(0, p_target(x) - p_draft(x)) / Z
  where Z is the normalization constant

  This ensures the final distribution matches the target model exactly.

Example:
  Draft:   p_draft(A) = 0.8, p_draft(B) = 0.2
  Target:  p_target(A) = 0.5, p_target(B) = 0.5

  If draft generates A:
    Accept prob = min(1, 0.5/0.8) = 0.625
    -> 62.5% chance to accept A

  If draft generates B:
    Accept prob = min(1, 0.5/0.2) = 1.0
    -> Always accept B

  Overall P(A) = 0.8 * 0.625 = 0.5 ✓
  Overall P(B) = 0.2 * 1.0 = 0.5 ✓
  -> Matches target distribution!
""")


def speculative_verify(draft_probs, target_probs, draft_tokens, temperature=1.0):
    """Speculative decoding acceptance/rejection algorithm"""
    batch_size, k, vocab_size = draft_probs.shape

    draft_p = draft_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)
    target_p = target_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)

    accept_rate = torch.clamp(target_p / draft_p.clamp_min(1e-8), max=1.0)
    random_vals = torch.rand_like(accept_rate)
    accepted_mask = random_vals < accept_rate

    accepted = torch.full((batch_size,), k, dtype=torch.long)
    for b in range(batch_size):
        for i in range(k):
            if not accepted_mask[b, i]:
                accepted[b] = i
                break

    return accepted, accepted_mask


print("\n[Demo] Acceptance/rejection verification")
print("-" * 60)
draft_probs = torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]])
target_probs = torch.tensor([[[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]]])
draft_tokens = torch.tensor([[0, 1, 2]])

print(f"Draft probs:   {draft_probs[0].tolist()}")
print(f"Target probs:  {target_probs[0].tolist()}")
print(f"Draft tokens:  {draft_tokens[0].tolist()}")

accepted, accepted_mask = speculative_verify(draft_probs, target_probs, draft_tokens)
print(f"Accepted mask: {accepted_mask[0].tolist()}")
print(f"First rejection at position: {accepted[0].item()}")


# ============================================================
# Part 4: Complete Speculative Decoding Implementation
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Complete Speculative Decoding Implementation")
print("=" * 60)


class SimpleDraftModel(nn.Module):
    """Simple draft model (smaller, faster)"""

    def __init__(self, vocab_size=100, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, vocab_size)

    def forward(self, ids):
        x = self.emb(ids)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    @torch.no_grad()
    def generate_k(self, ids, k=5, temperature=1.0):
        """Generate K tokens and return them with probabilities"""
        self.eval()
        all_tokens = []
        all_probs = []

        current_ids = ids.clone()
        for _ in range(k):
            logits = self(current_ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            all_tokens.append(next_token)
            all_probs.append(probs)
            current_ids = torch.cat([current_ids, next_token], dim=1)

        draft_tokens = torch.cat(all_tokens, dim=1)
        draft_probs = torch.stack(all_probs, dim=1)
        return draft_tokens, draft_probs


class SimpleTargetModel(nn.Module):
    """Simple target model (larger, more accurate)"""

    def __init__(self, vocab_size=100, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.fc3 = nn.Linear(dim, vocab_size)

    def forward(self, ids):
        x = self.emb(ids)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    @torch.no_grad()
    def verify_k(self, ids, draft_tokens, temperature=1.0):
        """Verify K draft tokens in one forward pass"""
        self.eval()
        full_ids = torch.cat([ids, draft_tokens], dim=1)
        logits = self(full_ids) / temperature
        probs = F.softmax(logits, dim=-1)
        target_probs = probs[:, ids.shape[1]-1:-1, :]
        bonus_probs = probs[:, -1, :].unsqueeze(1)
        return target_probs, bonus_probs


def speculative_decode(draft_model, target_model, ids, k=5, max_new_tokens=20,
                       temperature=1.0):
    """Complete speculative decoding loop"""
    generated = ids.clone()
    total_draft = 0
    total_accepted = 0

    while generated.shape[1] - ids.shape[1] < max_new_tokens:
        draft_tokens, draft_probs = draft_model.generate_k(
            generated, k=k, temperature=temperature
        )
        target_probs, bonus_probs = target_model.verify_k(
            generated, draft_tokens, temperature=temperature
        )

        accepted, accepted_mask = speculative_verify(
            draft_probs, target_probs, draft_tokens, temperature
        )

        total_draft += k
        total_accepted += accepted[0].item()

        for b in range(generated.shape[0]):
            n_accepted = accepted[b].item()
            new_tokens = draft_tokens[b, :n_accepted]

            if n_accepted < k:
                last_target_p = target_probs[b, n_accepted]
                last_draft_p = draft_probs[b, n_accepted]
                adjusted = torch.clamp(last_target_p - last_draft_p, min=0)
                adjusted = adjusted / adjusted.sum().clamp_min(1e-8)
                resampled = torch.multinomial(adjusted.unsqueeze(0), 1)
                new_tokens = torch.cat([new_tokens, resampled[0]])
            else:
                bonus_token = torch.multinomial(bonus_probs[b, 0].unsqueeze(0), 1)
                new_tokens = torch.cat([new_tokens, bonus_token[0]])

            generated = torch.cat([generated, new_tokens.unsqueeze(0)], dim=1)

    acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
    return generated, acceptance_rate


print("\n[Experiment] Speculative vs Autoregressive")
print("-" * 60)

torch.manual_seed(42)
vocab_size = 100
draft_model = SimpleDraftModel(vocab_size=vocab_size, dim=32)
target_model = SimpleTargetModel(vocab_size=vocab_size, dim=64)

input_ids = torch.randint(0, vocab_size, (1, 5))

start = time.time()
spec_output, acceptance_rate = speculative_decode(
    draft_model, target_model, input_ids, k=5, max_new_tokens=20
)
spec_time = time.time() - start

print(f"Speculative decoding (K=5):")
print(f"  Output length: {spec_output.shape[1]}")
print(f"  Acceptance rate: {acceptance_rate:.2%}")
print(f"  Time: {spec_time:.3f}s")


# ============================================================
# Part 5: Draft Model Selection
# ============================================================

print("\n" + "=" * 60)
print("Part 5: Draft Model Selection")
print("=" * 60)

print("""
Draft model requirements:
  1. Fast: Must be significantly faster than the target model
  2. Accurate: Higher acceptance rate = more speedup
  3. Same vocabulary: Must share the same tokenizer

Common choices:

1) Smaller version of the same family:
   Target: LLaMA-70B, Draft: LLaMA-7B
   -> Same architecture, same tokenizer
   -> Good acceptance rate (70-80%)

2) Distilled model:
   Train a small model to mimic the large model
   -> Higher acceptance rate (80-90%)
   -> But requires extra training

3) N-gram / n-gram-like:
   Use statistical patterns from training data
   -> Very fast, no GPU needed
   -> Lower acceptance rate (40-60%)

4) Same model, early exit:
   Use intermediate layer outputs as draft
   -> No extra model needed
   -> Moderate acceptance rate (60-70%)

Speedup formula:
  Speedup ≈ K * acceptance_rate / (1 + cost_ratio)

  K = draft length (typically 4-8)
  acceptance_rate = probability of accepting each token
  cost_ratio = draft_cost / target_cost (typically 0.05-0.1)

  Example: K=5, acceptance=0.8, cost_ratio=0.1
  Speedup ≈ 5 * 0.8 / (1 + 0.1) ≈ 3.6x
""")


# ============================================================
# Part 6: Advanced Techniques
# ============================================================

print("\n" + "=" * 60)
print("Part 6: Advanced Techniques")
print("=" * 60)

print("""
1) Adaptive K:
   - Dynamically adjust K based on acceptance rate
   - High acceptance -> Increase K
   - Low acceptance -> Decrease K
   - Better than fixed K

2) Tree-based speculation:
   - Draft model generates a tree of possible continuations
   - Target model verifies the entire tree in one pass
   - Higher throughput than linear speculation

3) Staged speculation:
   - Multiple draft models of increasing size
   - Small draft -> Medium draft -> Target
   - Cascading verification

4) Medusa heads:
   - Add multiple prediction heads to the target model
   - Each head predicts a future token
   - No separate draft model needed
   - Simpler deployment

5) Eagle:
   - Use the target model's own features for drafting
   - Train a lightweight draft head on top
   - Very high acceptance rate (90%+)
""")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Autoregressive Bottleneck
   - One forward pass per token
   - GPU underutilized (memory-bound)
   - Generation is slow

2. Speculative Decoding
   - Draft model generates K tokens (fast)
   - Target model verifies K tokens (one pass)
   - Accept/reject with adjusted resampling

3. Mathematical Guarantee
   - Output distribution is identical to target model
   - No quality loss, only speed gain
   - Typical speedup: 2-3x

4. Draft Model Selection
   - Same family smaller model (most common)
   - Distilled model (highest acceptance)
   - Early exit (no extra model)
   - N-gram (fastest, lowest acceptance)

5. Advanced Techniques
   - Adaptive K
   - Tree-based speculation
   - Medusa heads
   - Eagle

Next: Lesson 20 - Final Project: Complete MiniMind
""")


# ============================================================
# Deep Understanding: The "Draft-then-Verify" Philosophy
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: The 'Draft-then-Verify' Philosophy")
print("=" * 60)

print("""
[Analogy: Speculative Decoding = Fast-Track Review]
----------------------------------------------------

  Traditional generation:
    A senior professor writes every word personally
    -> High quality, but slow
    -> Professor is idle 90% of the time (thinking)

  Speculative decoding:
    A junior assistant quickly drafts K sentences
    The professor reviews all K sentences at once
    -> Accepts correct ones, revises incorrect ones
    -> Professor's time is fully utilized
    -> Output quality is identical to professor's own writing


[Why Speculative Decoding Works]
---------------------------------

  Key insight: LLM inference is memory-bound, not compute-bound

  Memory-bound means:
    Most time is spent loading weights from HBM to compute units
    Actual computation takes very little time
    -> Processing 1 token and 5 tokens takes almost the same time!

  Analogy:
    Like a truck delivering packages
    - Driving to the destination: 90% of time (loading weights)
    - Unloading packages: 10% of time (computation)
    - Delivering 1 package vs 5 packages: Almost the same time!

  Speculative decoding exploits this:
    Target model processes K+1 positions in one forward pass
    -> Cost ≈ 1 autoregressive step
    -> But potentially outputs K+1 tokens
    -> Speedup = K+1 / 1 = K+1 (in the best case)


[The Acceptance/Rejection Algorithm]
--------------------------------------

  For each draft token x_i:
    Accept with probability: min(1, p_target(x_i) / p_draft(x_i))

  Why does this preserve the target distribution?

  Proof sketch:
    P(output = x) = P(draft generates x) * P(accept x)
                  = p_draft(x) * min(1, p_target(x) / p_draft(x))
                  = min(p_draft(x), p_target(x))

    This is NOT yet p_target(x). The missing probability is
    compensated by the rejection resampling step:
    When rejected, sample from: max(0, p_target - p_draft) / Z

    Combined:
    P(output = x) = min(p_draft(x), p_target(x)) + p_reject * p_adjusted(x)

    After algebra: P(output = x) = p_target(x) ✓

  Intuition:
    - Tokens the target likes: Always accepted (or resampled to)
    - Tokens the target dislikes: Rejected and replaced
    - Net effect: Output follows target distribution


[Speedup Analysis]
-------------------

  Let α = acceptance rate per token

  Expected tokens per iteration:
    E[accepted] = α + α² + α³ + ... + α^K + α^K (bonus)
                = α(1 - α^K) / (1 - α) + α^K

  For K=5, α=0.8:
    E[accepted] ≈ 0.8 + 0.64 + 0.512 + 0.41 + 0.33 + 0.33 ≈ 3.0

  Cost per iteration:
    1 target forward + K draft forwards
    ≈ 1 + K * cost_ratio (cost_ratio ≈ 0.05-0.1)

  Speedup:
    E[accepted] / (1 + K * cost_ratio)
    ≈ 3.0 / (1 + 5 * 0.08) ≈ 2.1x

  Typical speedups:
    +------------------+----------+------------+
    | Draft Model      | α        | Speedup    |
    +------------------+----------+------------+
    | Same family 7B   | 70-80%   | 2.0-2.5x  |
    | Distilled        | 80-90%   | 2.5-3.0x  |
    | N-gram           | 40-60%   | 1.3-1.8x  |
    | Early exit       | 60-70%   | 1.5-2.0x  |
    +------------------+----------+------------+


[Choosing K (Draft Length)]
----------------------------

  K too small (K=1-2):
    Not enough speculation, limited speedup
    -> Overhead of verification is not amortized

  K too large (K=10+):
    Later tokens have very low acceptance rate
    -> Most draft tokens are wasted
    -> Draft model cost increases

  Optimal K:
    K ≈ -1 / ln(α) (where α is acceptance rate)

    α=0.8: K ≈ 4.5 -> K=4 or 5
    α=0.7: K ≈ 2.8 -> K=3
    α=0.9: K ≈ 9.5 -> K=8 or 9

  In practice:
    K=4-6 is a good default
    Adaptive K is even better


[Speculative Decoding vs Other Speedup Methods]
-------------------------------------------------

  +------------------+----------+----------+----------+
  | Method           | Speedup  | Quality  | Cost     |
  +------------------+----------+----------+----------+
  | Speculative      | 2-3x     | Exact    | Extra model|
  | Quantization     | 2-4x     | ~99%     | None     |
  | KV Cache        | 2-5x     | Exact    | Memory   |
  | Batch inference  | 5-10x    | Exact    | Memory   |
  | Distillation     | 3-5x     | ~95%     | Training |
  +------------------+----------+----------+----------+

  Combinations:
    Speculative + Quantization: 4-8x speedup
    Speculative + KV Cache: 4-10x speedup
    All combined: 10-20x speedup


[Practical Deployment Considerations]
--------------------------------------

  1. Memory:
    Need to load both draft and target models
    -> INT4 quantization for both helps

  2. Latency vs Throughput:
    Speculative decoding reduces latency per request
    But may reduce throughput (more compute per request)
    -> Best for latency-sensitive applications

  3. Batch size:
    Speculative decoding is most effective at batch_size=1
    At large batch sizes, GPU is already well-utilized
    -> Less benefit for high-throughput serving

  4. Draft model update:
    When target model is fine-tuned, draft model may need updating
    -> Acceptance rate drops if distributions diverge
    -> Periodically re-distill the draft model


[Medusa: Simplified Speculative Decoding]
------------------------------------------

  Problem: Managing two models is complex

  Medusa solution:
    Add multiple prediction heads to the target model itself
    - Head 0: Predict next token (original LM head)
    - Head 1: Predict token at position +1
    - Head 2: Predict token at position +2
    - ...

  Advantages:
    - No separate draft model
    - Simpler deployment
    - Only ~10% extra parameters

  Disadvantages:
    - Heads are less accurate than a dedicated draft model
    - Lower acceptance rate per position
    - But still achieves 2-2.5x speedup


[Eagle: Feature-Level Drafting]
---------------------------------

  Key insight:
    The target model's intermediate features contain rich information
    -> Use them to build a better draft

  Eagle approach:
    1. Run target model's first few layers to get features
    2. Use features + lightweight decoder to draft K tokens
    3. Target model verifies

  Advantages:
    - Very high acceptance rate (90%+)
    - Draft quality is much better than independent models
    - 3-4x speedup reported

  This is currently the SOTA approach for speculative decoding.
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Autoregressive Bottleneck
  Why is autoregressive generation slow? What is the bottleneck?

[Exercise 2] Acceptance Criterion
  Why is the acceptance criterion min(1, p_target/p_draft)?
  What happens if we always accept?

[Exercise 3] Distribution Preservation
  How does speculative decoding guarantee the output distribution
  matches the target model?

[Exercise 4] Draft Model Selection
  What are the pros and cons of using the same-family smaller model
  vs a distilled model as the draft?

[Exercise 5] K Selection
  If the acceptance rate is 0.7, what is the optimal K?
  What if it's 0.9?

[Exercise 6] Speedup Calculation
  Given: K=5, acceptance rate=0.8, cost_ratio=0.08
  Calculate the expected speedup.
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

print("\n[Exercise 1 Answer]")
print("  Autoregressive generation is slow because:")
print()
print("  1. One forward pass per token:")
print("    Each token requires loading all model weights from memory")
print("    -> 100 tokens = 100 weight loads")
print()
print("  2. Memory-bound, not compute-bound:")
print("    GPU compute units are idle most of the time")
print("    The bottleneck is weight loading, not computation")
print("    -> Processing 1 token vs 5 tokens: almost the same time")
print()
print("  3. Sequential dependency:")
print("    Each token depends on all previous tokens")
print("    Cannot parallelize across tokens")
print("    -> Inherent sequential bottleneck")

print("\n[Exercise 2 Answer]")
print("  Acceptance criterion: min(1, p_target(x) / p_draft(x))")
print()
print("  Why min(1, ...)?")
print("    If p_target > p_draft: ratio > 1, but probability can't exceed 1")
print("    -> Cap at 1 (always accept)")
print("    If p_target < p_draft: ratio < 1, accept with that probability")
print()
print("  What if we always accept?")
print("    Output distribution would match the DRAFT model, not the target")
print("    -> Quality degrades to draft model level")
print("    -> Defeats the purpose of using a target model")
print()
print("  What if we use p_target directly (no draft comparison)?")
print("    We wouldn't know which tokens to verify")
print("    The draft model's role is to PROPOSE candidates")
print("    The acceptance/rejection ensures we follow the TARGET distribution")

print("\n[Exercise 3 Answer]")
print("  Distribution preservation proof sketch:")
print()
print("  For each token position, the probability of outputting x is:")
print()
print("  P(output = x) = P(draft proposes x) × P(accept x)")
print("                 + P(reject) × P(resample x)")
print()
print("  = p_draft(x) × min(1, p_target(x)/p_draft(x))")
print("    + (1 - Σ min(p_draft, p_target)) × max(0, p_target(x) - p_draft(x))/Z")
print()
print("  After algebraic simplification:")
print("  P(output = x) = p_target(x)")
print()
print("  Key insight:")
print("    The acceptance step 'keeps' the overlap between draft and target")
print("    The rejection step 'adds back' the missing probability mass")
print("    Together, they exactly reproduce the target distribution")

print("\n[Exercise 4 Answer]")
print("  Same-family smaller model (e.g., LLaMA-7B for LLaMA-70B):")
print("    Pros:")
print("      + Same tokenizer, same architecture")
print("      + No extra training needed")
print("      + Readily available")
print("    Cons:")
print("      - Lower acceptance rate (70-80%)")
print("      - Still relatively large (7B is not tiny)")
print("      - More GPU memory needed")
print()
print("  Distilled model:")
print("    Pros:")
print("      + Higher acceptance rate (80-90%)")
print("      + Can be very small (1B or less)")
print("      + Less GPU memory")
print("    Cons:")
print("      - Requires extra training (distillation)")
print("      - Needs to be updated when target model changes")
print("      - Not always available for every model")
print()
print("  Recommendation:")
print("    Quick deployment: Same-family model")
print("    Maximum speedup: Distilled model")

print("\n[Exercise 5 Answer]")
print("  Optimal K ≈ -1 / ln(α)")
print()
print("  For α = 0.7:")
print("    K ≈ -1 / ln(0.7) ≈ -1 / (-0.357) ≈ 2.8")
print("    -> K = 3 is optimal")
print()
print("  For α = 0.9:")
print("    K ≈ -1 / ln(0.9) ≈ -1 / (-0.105) ≈ 9.5")
print("    -> K = 8 or 9 is optimal")
print()
print("  Intuition:")
print("    Higher acceptance rate -> Can speculate further")
print("    Lower acceptance rate -> Keep speculation short")
print()
print("  In practice:")
print("    Start with K=5 and adjust based on measured acceptance rate")

print("\n[Exercise 6 Answer]")
print("  Speedup calculation:")
print()
print("  Given: K=5, α=0.8, cost_ratio=0.08")
print()
print("  Step 1: Expected accepted tokens per iteration")
print("    E[accepted] = Σ(i=1 to K) α^i + α^K (bonus)")
print("    = 0.8 + 0.64 + 0.512 + 0.41 + 0.33 + 0.33")
print("    ≈ 3.02 tokens")
print()
print("  Step 2: Cost per iteration")
print("    Cost = 1 (target) + K × cost_ratio (draft)")
print("    = 1 + 5 × 0.08 = 1.4")
print()
print("  Step 3: Speedup")
print("    Speedup = E[accepted] / Cost")
print("    = 3.02 / 1.4 ≈ 2.16x")
print()
print("  This means speculative decoding generates tokens about 2.16x faster")
print("  than standard autoregressive decoding, with identical output quality")
