"""
Lesson 9: Training Loop — Teaching the Model to Speak
======================================================

We have the model and the data. How do we make the model "learn"?
The training loop is: repeatedly show data to the model, making it more accurate.

Core flow:
  for epoch in range(num_epochs):
      for batch in dataloader:
          1. Forward pass: compute predictions with current parameters
          2. Compute loss: gap between predictions and true labels
          3. Backward pass: compute gradients (how to adjust each parameter)
          4. Update parameters: fine-tune parameters along gradient direction

Key concepts:
  - Learning rate: magnitude of parameter adjustment per step; too large = unstable, too small = slow
  - Gradient clipping: prevent overly large gradients from crashing training
  - Learning rate scheduling: dynamically adjust learning rate during training

Run: python lessons_en/lesson09_training.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Reuse model from Lesson 8 (simplified)
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
# Step 1: Prepare Training Data
# ============================================================

print("=" * 60)
print("Experiment 1: Prepare Training Data")
print("=" * 60)

vocab_size = 200
seq_len = 32

torch.manual_seed(42)
pattern = torch.randint(0, vocab_size, (seq_len,))
train_data = pattern.repeat(100)

print(f"Training data length: {len(train_data)} tokens")
print(f"Sequence length: {seq_len}")
print(f"Vocabulary size: {vocab_size}")
print(f"Pattern: {pattern[:10].tolist()}... (repeated 100 times)")


# ============================================================
# Step 2: Simplest Training Loop
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: Simplest Training Loop")
print("=" * 60)

model = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Optimizer: AdamW, learning rate: 1e-3")
print(f"Expected initial loss: ln({vocab_size}) = {math.log(vocab_size):.4f}")
print()

losses = []
for step in range(50):
    start = torch.randint(0, len(train_data) - seq_len - 1, (1,)).item()
    batch = train_data[start:start + seq_len + 1].unsqueeze(0)

    input_ids = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if step % 10 == 0 or step == 49:
        print(f"Step {step:3d}: loss = {loss.item():.4f}")

print(f"\nLoss dropped from {losses[0]:.4f} to {losses[-1]:.4f}")
print("-> The model is learning! Loss is decreasing!")


# ============================================================
# Step 3: Gradient Clipping — Prevent Training Collapse
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: Gradient Clipping")
print("=" * 60)

model2 = MiniMindGPT(
    vocab_size=vocab_size, hidden_size=64, num_layers=2,
    num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=128,
)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

input_ids = torch.randint(0, vocab_size, (2, seq_len))
targets = torch.randint(0, vocab_size, (2, seq_len))

logits = model2(input_ids)
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
loss.backward()

total_norm = 0
for p in model2.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm before clipping: {total_norm:.4f}")

max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm)

total_norm_after = 0
for p in model2.parameters():
    if p.grad is not None:
        total_norm_after += p.grad.data.norm(2).item() ** 2
total_norm_after = total_norm_after ** 0.5

print(f"Gradient norm after clipping: {total_norm_after:.4f} (max_norm={max_norm})")
print(f"\nGradient clipping ensures each step's update isn't too aggressive, preventing training collapse")


# ============================================================
# Step 4: Learning Rate Scheduling
# ============================================================

print("\n" + "=" * 60)
print("Experiment 4: Learning Rate Scheduling — Cosine Annealing")
print("=" * 60)

class CosineScheduler:
    """Cosine annealing learning rate scheduler"""

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

print("Learning rate changes:")
print(f"{'Step':>6} | {'LR':>10}")
print("-" * 20)
for step in [0, 5, 10, 25, 50, 75, 100]:
    lr = scheduler.get_lr(step)
    print(f"  {step:>4} | {lr:>10.6f}")

print("\nWarmup phase (steps 0-10): LR linearly increases from 0 to max")
print("Decay phase (steps 10-100): LR decreases from max to min following cosine curve")


# ============================================================
# Step 5: Complete Training Loop
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5: Complete Training Loop (with all techniques)")
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

print(f"Configuration:")
print(f"  Optimizer: AdamW (weight_decay=0.01)")
print(f"  Learning rate: 5e-4 -> 1e-5 (cosine annealing, warmup=5 steps)")
print(f"  Gradient clipping: max_norm={max_grad_norm}")
print(f"  Total steps: {num_steps}")
print()

losses_full = []
for step in range(num_steps):
    lr = scheduler4.step(step)

    starts = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,))
    batch = torch.stack([train_data[s:s + seq_len + 1] for s in starts])
    input_ids = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model4(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer4.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model4.parameters(), max_grad_norm)

    optimizer4.step()

    losses_full.append(loss.item())
    if step % 20 == 0 or step == 99:
        print(f"Step {step:3d}: loss={loss.item():.4f}, lr={lr:.6f}, grad_norm={grad_norm:.4f}")

print(f"\nLoss: {losses_full[0]:.4f} -> {losses_full[-1]:.4f}")
print("Training helped the model learn the patterns in the data!")


# ============================================================
# Step 6: Validate Training Results
# ============================================================

print("\n" + "=" * 60)
print("Experiment 6: Validate Training Results")
print("=" * 60)

model4.eval()
with torch.no_grad():
    test_input = pattern[:8].unsqueeze(0)
    logits = model4(test_input)
    predictions = logits.argmax(dim=-1)

print(f"Input:          {test_input[0].tolist()}")
print(f"Predictions:    {predictions[0].tolist()}")
print(f"True next word: {pattern[1:9].tolist()}")

correct = (predictions[0] == pattern[1:9]).sum().item()
print(f"Correct: {correct}/8")

with torch.no_grad():
    full_input = pattern.unsqueeze(0)
    logits = model4(full_input)
    shift_logits = logits[:, :-1, :]
    shift_labels = pattern[1:].unsqueeze(0)
    loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))
    perplexity = math.exp(loss.item())

print(f"\nPerplexity: {perplexity:.2f}")
print(f"Random model perplexity: {vocab_size} (= vocabulary size)")
print(f"Lower perplexity = more accurate model predictions")


# ============================================================
# Step 7: Key Training Concepts Summary
# ============================================================

print("\n" + "=" * 60)
print("Experiment 7: Key Training Concepts")
print("=" * 60)

print("""
1. Forward pass: input -> model -> logits -> loss
2. Backward pass: loss -> compute gradient for each parameter
3. Parameter update: param = param - lr * gradient
4. Gradient clipping: prevent overly large gradients from crashing training
5. Learning rate scheduling: warmup + cosine decay
6. Weight decay: prevent parameters from growing too large, improve generalization

MiniMind training configuration:
  Optimizer: AdamW (weight_decay=0.01)
  Learning rate: 5e-4 (cosine annealing)
  Batch size: 64
  Sequence length: 512
  Gradient clipping: max_norm=1.0
  Training steps: ~50K steps
""")


# ============================================================
# Deep Dive: The Essence of the Training Loop
# ============================================================
print("\n" + "=" * 60)
print("Deep Dive: The Essence of the Training Loop")
print("=" * 60)

print("""
[Analogy: Training is Like "Teaching a Student"]
------------------------------------------------
  Model = Student (blank slate)
  Training data = Textbook
  Loss = Exam score (lower is better)
  Gradient = Error analysis (tells the student what went wrong)
  Optimizer = Study method (how to improve)

  Process:
    1. Teacher lectures (forward) -> Student answers questions
    2. Grade the exam (loss)
    3. Analyze mistakes (backward)
    4. Student improves (parameter update)
    5. Repeat until mastery


[Diagram: Training Loop]
------------------------

  +------------------------------------+
  |       Training Loop (repeat N)      |
  |                                    |
  |  +----------+  +----------+        |
  |  | Forward  |->|Loss calc |        |
  |  +-----+----+  +----+-----+        |
  |        |            |              |
  |        v            |              |
  |  +----------+       |              |
  |  | Backward |<------+              |
  |  |(gradient)|                      |
  |  +-----+----+                      |
  |        |                           |
  |        v                           |
  |  +----------+                      |
  |  | Optimizer| (AdamW etc.)         |
  |  | update   |                      |
  |  +-----+----+                      |
  |        |                           |
  |        +---- back to Forward ----+  |
  +------------------------------------+


[Common Reasons for Loss Not Decreasing]
----------------------------------------
  1. Learning rate too large:
     Symptom: Loss oscillates or even diverges
     Solution: Reduce lr (e.g., 1e-4 -> 1e-5)

  2. Learning rate too small:
     Symptom: Loss barely moves
     Solution: Increase lr, or use larger warmup

  3. Data issues:
     Symptom: Training loss also doesn't decrease
     Solution: Check data, verify initial loss value

  4. Model too small:
     Symptom: Loss decreases but plateaus at a high value
     Solution: Use a larger model

  5. Not enough training:
     Symptom: Loss is still decreasing
     Solution: Continue training

  6. Batch size too large:
     Symptom: Loss decreases but poor generalization
     Solution: Reduce batch size


[Intuitive Understanding of LR Scheduling]
------------------------------------------
  Analogy: Learning to ride a bike

  Warmup phase (beginning):
    LR slowly increases from 0
    Reason: Model just started, can't take big steps
    Analogy: Just learning to ride, go slow to find balance

  Stable phase (middle):
    LR stays relatively large
    Analogy: Know the basics, try boldly

  Decay phase (late):
    LR gradually decreases
    Analogy: Already skilled, refine the details

  Curve (ideal):

  lr|   /\\
     |  /  \\___________
     | /              \\_______
     |/                       \\____
     +------------------------------ step
       warmup  cosine decay


[Gradient Accumulation]
-----------------------
  Problem: Not enough GPU memory for large batch
  Solution: Simulate large batch

  Real batch = 4
  Accumulation steps = 4
  Effective batch = 16

  Steps:
    step 1: forward + backward (accumulate gradient)
    step 2: forward + backward (accumulate gradient)
    step 3: forward + backward (accumulate gradient)
    step 4: forward + backward (accumulate gradient)
    step 5: optimizer.step() (update with accumulated gradients)

  -> Simulate large batch with small GPU memory, equivalent training effect
""")

# [NEW] End-to-End Training → Generation Demo
print("\n" + "-" * 50)
print("[End-to-End Demo: Train a TinyGPT and Generate Text]")
print("-" * 50)

# TinyGPT: a minimal GPT model for demonstration
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, n_heads=2, n_layers=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(64, embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=embed_dim*4, dropout=0.0, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    def generate(self, idx, max_new_tokens=20, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -64:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Train on simple data
tiny_vocab_size = 50
tiny_model = TinyGPT(tiny_vocab_size)
tiny_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

# Simple training data: sequences of token IDs
torch.manual_seed(42)
train_data = torch.randint(0, tiny_vocab_size, (32, 16))  # 32 samples, length 16

print("Training TinyGPT for 100 steps...")
for step in range(100):
    x = train_data[:, :-1]
    y = train_data[:, 1:]
    logits, loss = tiny_model(x, y)
    tiny_optimizer.zero_grad()
    loss.backward()
    tiny_optimizer.step()
    if (step + 1) % 50 == 0:
        print(f"  Step {step+1}: loss = {loss.item():.4f}")

# Generate text
print("\nGenerating text from TinyGPT:")
start_ids = torch.randint(0, tiny_vocab_size, (1, 4))
generated = tiny_model.generate(start_ids, max_new_tokens=10, temperature=0.8)
print(f"  Input IDs: {start_ids.tolist()[0]}")
print(f"  Generated IDs: {generated.tolist()[0]}")
print("""
  This demonstrates the complete end-to-end workflow:
    1. Define model architecture (TinyGPT)
    2. Prepare training data (token ID sequences)
    3. Train with forward → loss → backward → update loop
    4. Generate new text by sampling from the trained model
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Loss Not Decreasing
  Training loss is NaN, what could be the reasons?

[Exercise 2] Gradient Clipping
  Why do we need gradient clipping? What does max_norm=1.0 mean?

[Exercise 3] Batch Size and Learning Rate
  If batch size increases 4x, how should learning rate change?

[Exercise 4] Gradient Accumulation
  Real batch=2, target batch=32, how many accumulation steps?

[Exercise 5] Overfitting vs Underfitting
  Training loss keeps decreasing, validation loss decreases then increases
  What is this phenomenon? How to solve it?

[Exercise 6] AdamW vs SGD
  Briefly compare the pros and cons of AdamW and SGD for LLM training
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Possible reasons for Loss = NaN:")
print()
print("  1. Learning rate too large:")
print("    - Gradient explosion, parameters become inf/nan")
print("    - Solution: Reduce lr by 10x")
print()
print("  2. Data contains NaN:")
print("    - Some token IDs exceed vocab_size")
print("    - Numerical computation produces 0/0")
print("    - Solution: Clean the data")
print()
print("  3. Numerical overflow:")
print("    - Attention scores too large, softmax produces NaN")
print("    - Solution: Add gradient clipping, or scale the loss")
print()
print("  4. Improper model initialization:")
print("    - Some weights initialized to 0 or extreme values")
print("    - Solution: Check initialization")
print()
print("  Debugging tips:")
print("    - Try lr=1e-6 first, see if loss is stable")
print("    - Single-step debug, print intermediate values")
print("    - Monitor gradient norm")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Reasons for gradient clipping:")
print()
print("  1. Prevent gradient explosion:")
print("    - In deep networks, gradients can grow exponentially")
print("    - One large gradient update can push the model into a bad region")
print("    - Loss suddenly spikes, hard to recover")
print()
print("  2. max_norm=1.0 meaning:")
print("    - If the L2 norm of all parameter gradients > 1.0")
print("    - Scale the gradient down so norm = 1.0")
print("    - Doesn't change direction, only reduces magnitude")
print()
print("  Formula: grad = grad * min(1, max_norm / ||grad||)")
print()
print("  Experimental comparison:")
print("    Without clipping: loss occasionally spikes, unstable")
print("    With clipping: loss decreases smoothly, training stable")

# Demo
torch.manual_seed(42)
m = nn.Linear(4, 4)
x = torch.randn(1, 4)
y = torch.randn(1, 4)
loss = ((m(x) - y) ** 2).sum()
loss.backward()
grad_norm = sum(p.grad.norm() ** 2 for p in m.parameters()) ** 0.5
print(f"  Example gradient norm: {grad_norm:.4f}")
print(f"  After clipping to max_norm=1.0: 1.0000")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Rule of thumb: Linear scaling rule")
print()
print("  Original: bs=32,  lr=1e-4")
print("  New:      bs=128, lr=4e-4  (4x)")
print()
print("  Reason:")
print("    - Large batch gradients are more accurate 'averages'")
print("    - Don't need as conservative a learning rate")
print("    - But too large can also be unstable")
print()
print("  In practice, sqrt scaling is often used:")
print("    Batch increases 4x, lr increases 2x")
print("    More conservative, more stable")
print()
print("  Also needed:")
print("    - Warmup steps should increase proportionally")
print("    - Because larger batch is equivalently 'faster'")

# Exercise 4
print("\n[Exercise 4 Answer]")
real_batch = 2
target_batch = 32
accum_steps = target_batch // real_batch
print(f"  Real batch = {real_batch}")
print(f"  Target batch = {target_batch}")
print(f"  Accumulation steps = {target_batch} / {real_batch} = {accum_steps} steps")
print()
print("  Code flow:")
print("    for step in range(accum_steps):")
print("        loss = model(batch)")
print("        loss = loss / accum_steps  # scale")
print("        loss.backward()           # accumulate gradients")
print("    optimizer.step()             # update parameters")
print("    optimizer.zero_grad()")
print()
print("  Notes:")
print("    - Loss must be divided by accum_steps to be equivalent to large batch")
print("    - Only call optimizer.step() after the last step")
print("    - BatchNorm works better with large batch (LLMs use RMSNorm, no issue)")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Phenomenon: Overfitting")
print()
print("  Symptoms:")
print("    Training loss:   ________ (keeps decreasing)")
print("    Validation loss: ____/\\___ (decreases then increases)")
print("                           ^")
print("                    Overfitting starts here")
print()
print("  Causes:")
print("    - Model learned the 'noise' in training data")
print("    - Training data is too specific, validation data is different")
print("    - Model has poor generalization")
print()
print("  Solutions:")
print("    1. More data (most effective)")
print("    2. Smaller model (reduce capacity)")
print("    3. Regularization:")
print("       - Dropout (randomly drop during training)")
print("       - Weight Decay (penalize large parameters)")
print("       - Data augmentation")
print("    4. Early Stopping:")
print("       - Stop training when validation loss starts increasing")
print("    5. Simplify the task (if possible)")
print()
print("  LLM situation:")
print("    - Data volume is huge, rarely overfits")
print("    - Main concern is 'underfitting' (not enough training)")
print("    - But small models on small data can still overfit")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  SGD (Stochastic Gradient Descent):")
print("    Pros:")
print("      - Simple, low memory usage")
print("      - Stable training")
print("      - Good generalization")
print("    Cons:")
print("      - Slow convergence")
print("      - Can stall at saddle points")
print("      - Sensitive to learning rate")
print()
print("  AdamW (Adam + Weight Decay):")
print("    Pros:")
print("      - Adaptive learning rate (different lr per parameter)")
print("      - Fast convergence")
print("      - Less sensitive to hyperparameters")
print("    Cons:")
print("      - Higher memory usage (stores first and second moments)")
print("      - Slightly worse generalization than SGD")
print("      - Occasionally unstable training")
print()
print("  Practical choice:")
print("    LLM pretraining: AdamW (mainstream, fast convergence)")
print("    Vision fine-tuning: SGD or AdamW")
print("    Small datasets: AdamW easily overfits, SGD more stable")
print()
print("  Advanced:")
print("    - LION (2023): Less memory, faster")
print("    - Muon (2024): Uses Newton-Schulz orthogonalization, better results")
print("    - These are covered in MiniMind Lesson 12")


# ============================================================
# Lesson Summary
# ============================================================
print("=" * 60)
print("Lesson Summary")
print("=" * 60)
print("""
1. Training loop = forward pass + compute loss + backward pass + update parameters
2. Decreasing loss = model is learning
3. Gradient clipping prevents training collapse
4. Learning rate scheduling: warmup + cosine decay
5. Perplexity measures model quality, lower is better

Complete training flow:
  Data -> Model -> logits -> loss -> gradients -> update parameters -> repeat
  "Hello" -> Model -> Predict "world" -> Wrong -> Adjust -> More accurate next time

Next -> lesson10_generation.py: Making the model write word by word
""")
