"""
Lesson 15: LoRA — Fine-Tuning Large Models with Minimal Parameters
==================================================================

The problem with Full Fine-Tuning:
  Assume a 7B parameter model → Adam optimizer needs 3x parameters (m, v, grad)
  Memory = 4 bytes × 7B × 3 × 2 = 168GB (fp32)
  Impossible to fine-tune on consumer GPUs!

LoRA (Low-Rank Adaptation) solution:
  Freeze original weights W
  Inject two low-rank matrices A and B
  Only train A and B (very few parameters!)

[Why Does Low-Rank Work? — Intuition]

  Full fine-tuning = Renovating a house: you can change all walls, floors, ceilings (4096 directions)
  LoRA = Renovating a house: only repaint walls and change curtains (8 directions is enough)

  Research shows that during fine-tuning, the weight change ΔW has "effective rank" of only 8-64.
  That means in a 4096-dimensional space, only 8-64 directions are actually changing!

  Why? Because fine-tuning adjusts an already-trained model, not training from scratch.
  The model already knows the language — fine-tuning only needs small adjustments.
  These adjustments happen in a low-dimensional subspace.

  Analogy:
    A portrait photo needs editing → you adjust brightness, contrast, saturation (3 knobs)
    You don't need to repaint every pixel from scratch (millions of knobs)
    LoRA = finding the right "knobs" to adjust

This lesson covers:
  1. What is low-rank decomposition
  2. Mathematical principles of LoRA
  3. PyTorch implementation of LoRA
  4. How to use LoRA in MiniMind

Run: python lessons_en/lesson15_lora.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: The Dilemma of Full Fine-Tuning
# ============================================================

print("=" * 60)
print("Part 1: The Dilemma of Full Fine-Tuning")
print("=" * 60)

print("""
Full Fine-Tuning requires updating all parameters.

Assume a 7B parameter model:
  7B × 4 bytes (fp32) = 28GB  weights
  7B × 4 bytes = 28GB           gradients
  7B × 4 bytes × 2 = 56GB      Adam state (m, v)
  7B × 4 bytes = 28GB           activations (approximate)
  ─────────────────
  Total ≈ 140GB

  Top-tier A100 GPU 80GB × 2 = only 160GB! Can't train!

Purpose of Fine-Tuning:
  Adapt to new tasks (QA, dialogue, code, ...)
  Solve "catastrophic forgetting"
  Training cost too high, what to do?
""")

print("\n[Experiment] Training memory for different model sizes")
print("-" * 60)
print(f"{'Model':<12}{'Params':<12}{'Weights':<10}{'Grad+Optim':<15}{'Total':<10}")
print("-" * 60)

for params_b in [0.1, 1, 7, 70, 175]:
    weight_gb = params_b * 4
    train_state_gb = params_b * 12
    total = weight_gb + train_state_gb
    print(f"{params_b}B{'':<10}{params_b:.1f}B{'':<6}{weight_gb:.0f}GB{'':<4}{train_state_gb:.0f}GB{'':<10}{total:.0f}GB")


# ============================================================
# Part 2: LoRA's Core Idea
# ============================================================

print("\n" + "=" * 60)
print("Part 2: LoRA's Core Idea — Low-Rank Decomposition")
print("=" * 60)

print("""
LoRA's key observation:
  Pre-trained model's weight change ΔW is typically "low-rank"!
  → No need for a d×d update matrix
  → Only need two small matrices: d×r and r×d

Mathematical principle:
  Original: y = W·x            (W is d×d matrix)
  LoRA:     y = W·x + (B·A)·x

  Where:
    W: Original weights (frozen)  - d×d
    A: Down projection (trained)  - d×r  (r << d)
    B: Up projection (trained)    - r×d

  Parameter comparison:
    Original:  d × d
    LoRA:      2 × d × r

  When d=1024, r=8:
    Original:   1,048,576 parameters
    LoRA:       16,384 parameters (64x fewer!)

Analogy:
  Fine-tuning a photo:
  - Full fine-tuning = Redraw the entire photo
  - LoRA = Only apply "tiny modifications"
  - These modifications can be expressed in low dimensions
""")


# ============================================================
# Part 3: LoRA Mathematical Details
# ============================================================

print("\n" + "=" * 60)
print("Part 3: LoRA Mathematical Details")
print("=" * 60)

print("""
LoRA initialization:
  A: Kaiming uniform initialization (similar to nn.Linear)
  B: Initialized to 0 → At training start, BA = 0
  → Training starting point = original model! Ensures stability

Training objective:
  min ||ΔW - BA||²  where BA is the low-rank approximation

Forward pass:
  output = W·x + (B·A)·x · (α/r)

  α/r is the scaling factor:
    α: LoRA scaling parameter (commonly 16 or 32)
    r: rank
    α/r: Keeps update magnitude independent of r
""")

print("\n[Experiment] Low-rank decomposition parameter count")
print("-" * 60)

d = 1024
r = 8

W = torch.randn(d, d)
A = torch.randn(d, r)
B = torch.randn(r, d)

orig_params = d * d
lora_params = d * r + r * d

print(f"Hidden dimension: d = {d}")
print(f"LoRA rank:        r = {r}")
print(f"\nOriginal weight W:  {d} × {d} = {orig_params:,} parameters")
print(f"LoRA matrix A:      {d} × {r} = {d*r:,} parameters")
print(f"LoRA matrix B:      {r} × {d} = {r*d:,} parameters")
print(f"LoRA total params:  {lora_params:,}")
print(f"\nParameter reduction: {(1 - lora_params/orig_params)*100:.2f}%")
print(f"Compression ratio:  {orig_params/lora_params:.0f}x")

print("\n[Demo] Verify BA approximates a simple low-rank matrix")
print("-" * 60)
d_demo = 32
r_demo = 4
U = torch.randn(d_demo, r_demo)
V = torch.randn(d_demo, r_demo)
target = U @ V.T

A = torch.randn(r_demo, d_demo) * 0.01
B = torch.zeros(d_demo, r_demo)
BA = B @ A

lr = 0.001
for step in range(100):
    error = target - BA
    grad_A = -B.T @ error
    grad_B = -error @ A.T
    A = A - lr * grad_A
    B = B - lr * grad_B
    BA = B @ A
    if step % 20 == 0:
        print(f"Step {step}: Frobenius error = {(target - BA).norm():.4f}")


# ============================================================
# Part 4: Complete LoRA Implementation
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Complete LoRA Implementation")
print("=" * 60)


class LoRALinear(nn.Module):
    """Linear layer with LoRA"""

    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    def freeze_original(self):
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def unfreeze_lora(self):
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True


print("\n[Experiment] Demo LoRA Linear layer")
print("-" * 60)
lora_layer = LoRALinear(256, 256, r=8, alpha=16)

x = torch.randn(2, 10, 256)
y_init = lora_layer(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y_init.shape}")

total = sum(p.numel() for p in lora_layer.parameters())
trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
print(f"\nTotal params:     {total:,}")
print(f"Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")


# ============================================================
# Part 5: Merging LoRA Weights
# ============================================================

print("\n" + "=" * 60)
print("Part 5: Merging LoRA Weights (for Deployment)")
print("=" * 60)

print("""
After training, LoRA weights can be merged into original weights:
  W_merged = W + B·A · (α/r)

After merging:
  - Model size unchanged
  - Inference speed unchanged
  - No longer need LoRA module
  - Can deploy like a regular model
""")

def merge_lora(lora_layer: LoRALinear):
    delta_w = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
    lora_layer.weight.data = lora_layer.weight.data + delta_w
    lora_layer.lora_A.data.zero_()
    lora_layer.lora_B.data.zero_()


print("\n[Demo] Merge LoRA weights")
print("-" * 60)
lora_layer2 = LoRALinear(64, 64, r=4, alpha=8)
W_before = lora_layer2.weight.data.clone()
merge_lora(lora_layer2)
W_after = lora_layer2.weight.data.clone()
print(f"Weight norm before merge: {W_before.norm():.4f}")
print(f"Weight norm after merge:  {W_after.norm():.4f}")
print(f"Difference: {(W_after - W_before).norm():.4f} (LoRA's contribution)")


# ============================================================
# Part 6: Using LoRA in MiniMind
# ============================================================

print("\n" + "=" * 60)
print("Part 6: Using LoRA in MiniMind")
print("=" * 60)

print("""
MiniMind's LoRA fine-tuning workflow:

1. Load pre-trained model
2. Freeze all parameters
3. Add LoRA to attention layers (q_proj, v_proj)
4. Only train LoRA parameters
5. Merge weights after training

Code example (from MiniMind):
```python
from peft import LoraConfig, get_peft_model, TaskType

# Load model
model = AutoModelForCausalLM.from_pretrained("minimind")

# LoRA config
lora_config = LoraConfig(
    r=8,                    # rank
    lora_alpha=16,          # scaling
    target_modules=["q_proj", "v_proj"],  # which layers to apply
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Inject LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 1.5M || all params: 26M || trainable%: 5.77%

# Train (only update LoRA parameters)
trainer.train()

# Merge weights
model = model.merge_and_unload()
model.save_pretrained("minimind-lora")
```

Layers where LoRA can be applied:
  ✓ q_proj, k_proj, v_proj, o_proj (Attention)
  ✓ gate_proj, up_proj, down_proj (FFN)
  ✗ embedding, lm_head (usually not recommended)

Rank selection:
  r=4:  Very small tasks, save more parameters
  r=8:  Balanced choice (most common)
  r=16: Complex tasks, stronger expressiveness
  r=32+: Approaching full fine-tuning
""")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Cost of Full Fine-Tuning
   - 7B model: 140GB memory
   - 175B model: 3.5TB memory
   - Consumer GPUs can't train

2. LoRA's Core Idea
   - Weight change ΔW is low-rank
   - Use two small matrices BA to approximate ΔW
   - 100x-1000x fewer parameters

3. LoRA Implementation
   - Freeze original weights W
   - Inject trainable matrices A (r×d) and B (d×r)
   - Scaling factor α/r
   - Training starting point = original model (B=0)

4. Deployment Optimization
   - Merge weights after training
   - Model size unchanged
   - Inference speed unchanged

5. In MiniMind
   - Use with peft library
   - Typically apply LoRA only to q_proj, v_proj
   - rank=8 is the common choice

Next: Lesson 16 - YaRN Length Extrapolation
""")


# ============================================================
# Deep Understanding: LoRA's "Small but Beautiful"
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: LoRA's 'Small but Beautiful'")
print("=" * 60)

print("""
[Analogy: LoRA = Altering Clothes Without Adding Fabric]
──────────────────────

  Full fine-tuning:
    Take the entire garment apart and remake it
    → Big changes, lots of labor, high cost

  LoRA:
    Add a thin "lining" over the original garment
    → Looks the same, but can change style
    → Less labor, lower cost, easy to swap

  Mathematically:
    Original garment (W) stays unchanged
    Lining (BA) adjusts the fit
    Actual fit: W + BA

  Advantages:
    - Lining can be removed, restore original look
    - Multiple linings for different occasions
    - Extremely low total cost


[Mathematical Essence of LoRA]
─────────────────────────────

  Low-rank assumption (Intrinsic Dimension):
    Model fine-tuning doesn't need to change many directions
    Although W has d×d parameters
    The effective "directions of change" are few (rank is low)

  Verification:
    Take d=4096, theoretically 4096 dimensions of change
    In practice only 8-64 dimensions suffice
    → 99% of 'change directions' are redundant

  Mathematical expression:
    ΔW ∈ R^{d × d}, but rank(ΔW) ≤ r
    ΔW = B @ A, where B ∈ R^{d × r}, A ∈ R^{r × d}
    Parameter count: d×r + r×d = 2dr, much less than d²


[Diagram: LoRA Structure]
──────────────────────

  Input x ──┬────────── [W] ──────────┐
             │   (frozen)               ↓
             └────────── [B] ─→ [A] ─── + → Output
                  (trained)      (trained)

  Details:
    A: [r, d]  Down-project first (d → r)
    B: [d, r]  Up-project after (r → d)
    BA: [d, d]  Simulates ΔW (but rank ≤ r)

  Where:
    A initialized with Gaussian
    B initialized to 0
    → At training start, LoRA output is 0, model behavior unchanged


[LoRA Variants]
──────────────

  ┌────────────┬──────────────────┬──────────────────┐
  │ Variant    │ Description      │ Use Case         │
  ├────────────┼──────────────────┼──────────────────┤
  │ LoRA       │ Standard         │ General          │
  │ QLoRA      │ 4-bit + LoRA     │ Very tight memory│
  │ DoRA       │ Decompose into direction+magnitude │ More stable│
  │ AdaLoRA    │ Adaptive rank    │ Efficient        │
  │ LoRA+      │ A, B different learning rates │ Faster convergence│
  │ rsLoRA     │ rank-stabilized  │ Training stability│
  └────────────┴──────────────────┴──────────────────┘

  Mainstream choices:
    - Single GPU fine-tuning 7B: QLoRA
    - Normal fine-tuning: LoRA (rank=16)
    - Experimental: DoRA / AdaLoRA


[LoRA vs Full Fine-Tuning Comparison]
───────────────────────────────────

  ┌────────────┬─────────────┬──────────────┐
  │ Metric     │ Full FT     │ LoRA         │
  ├────────────┼─────────────┼──────────────┤
  │ Memory (7B)│ 60+ GB      │ 16 GB        │
  │ Train speed│ 1x          │ 1.2x (faster!)│
  │ Parameters │ 7B          │ 4M (0.05%)   │
  │ Storage/task│ 14 GB      │ 16 MB        │
  │ Inference  │ 1x          │ 1x (mergeable)│
  │ Performance│ 100%        │ 95-99%       │
  └────────────┴─────────────┴──────────────┘

  Key insights:
    - LoRA saves 70%+ memory
    - LoRA training is slightly faster (fewer params to update)
    - Performance loss typically < 5%
    - After merging, no inference overhead


[Rank Selection]
──────────────

  Smaller rank:
    - Fewer training parameters
    - Less memory
    - Limited expressiveness
    - Suitable for: Simple tasks, small data

  Larger rank:
    - More training parameters
    - More memory
    - Stronger expressiveness
    - Suitable for: Complex tasks, large data

  Empirical values:
    rank=4:  Minimal tasks (sentiment classification)
    rank=8:  Common tasks (recommended)
    rank=16: Complex tasks (dialogue)
    rank=32: Highly customized
    rank=64: Approaching full fine-tuning

  Experiments (LLaMA-7B):
    rank=8:  94% performance
    rank=16: 97%performance
    rank=32: 99% performance
    rank=64: 99.5% performance
    → Diminishing marginal returns


[LoRA Application Scenarios]
──────────────────────────

  1. Instruction Tuning
    Base model → Add instruction lining → Obedient assistant

  2. Personalization
    Base model → Add different linings → Different styles

  3. Multi-task
    1 base model + N LoRA sets
    → Save (N-1)x storage

  4. Continual Learning
    Learn new tasks, old task linings untouched
    → Avoid catastrophic forgetting

  5. Deployment Optimization
    Use LoRA during training, merge for inference
    → No inference speed loss


[QLoRA: Extreme Memory Optimization]
──────────────────────────────────

  4-bit quantized base + LoRA:
    Base model: 4-bit storage
    LoRA parameters: FP16 training
    → Memory savings to the extreme

  Configuration (LLaMA-65B):
    Full fine-tuning: 780 GB
    LoRA:  240 GB
    QLoRA: 48 GB (single GPU possible!)

  Performance:
    QLoRA: ~99% of LoRA performance
    Almost no loss
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] LoRA Parameter Calculation
  7B model, hidden=4096, rank=8
  How many parameters does LoRA add? What percentage of total?

[Exercise 2] Rank Selection
  Task: 100 data samples, simple classification
  What rank do you recommend?

[Exercise 3] LoRA Target Layers
  Which layers typically get LoRA? Why?

[Exercise 4] LoRA Initialization
  Why is A initialized with Gaussian and B initialized to 0?

[Exercise 5] QLoRA Memory
  How much memory does a 65B model need with QLoRA?

[Exercise 6] LoRA Merging
  How to merge LoRA into the original model after training?
  Can it be unmerged after merging?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
d = 4096
r = 8
lora_per_layer = 2 * d * r
print(f"  hidden_size = {d}, rank = {r}")
print(f"  Per-layer LoRA params: 2 × d × r = 2 × {d} × {r} = {lora_per_layer:,}")
print()
total_7b = 7e9
print(f"  7B model total params: ~{total_7b/1e9:.0f}B")
print(f"  If applying LoRA to 32 layers' q_proj:")
total_lora = lora_per_layer * 32
ratio = total_lora / total_7b * 100
print(f"  LoRA total params: {lora_per_layer:,} × 32 = {total_lora:,} = {total_lora/1e6:.1f} M")
print(f"  Percentage of total: {ratio:.3f}%")
print()
print(f"  → Only need to train ~0.05% of parameters")
print(f"  → Save 99.95% of optimizer state")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Simple task, little data → small rank")
print()
print("  Recommendation: rank=4 or rank=8")
print()
print("  Reason:")
print("    - 100 samples can express limited 'change'")
print("    - Too large a rank leads to overfitting")
print("    - Small rank trains fast, less overfitting")
print()
print("  Experiments:")
print("    100 samples + rank=64 → Overfitting")
print("    100 samples + rank=8  → Good generalization")
print()
print("  Empirical formula:")
print("    rank ≈ sqrt(data_count / 1000)")
print("    100 samples  → rank=1 (or slightly more)")
print("    1K samples   → rank=1-4")
print("    10K samples  → rank=8-16")
print("    100K+ samples → rank=16-64")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Common LoRA target layers:")
print()
print("  LLaMA style:")
print("    target_modules = ['q_proj', 'v_proj']")
print("    → Most common, balances effectiveness and efficiency")
print()
print("  More comprehensive:")
print("    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
print("    → Better results, but more parameters")
print()
print("  Full attention + FFN:")
print("    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',")
print("                       'gate_proj', 'up_proj', 'down_proj']")
print("    → Close to full fine-tuning effectiveness")
print()
print("  Guidelines:")
print("    Simple task: Only q_proj, v_proj")
print("    Complex task: Add o_proj")
print("    Very complex: Add all")
print()
print("  Layers NOT to apply LoRA:")
print("    - embedding (changes break word vectors)")
print("    - lm_head (same as embedding)")
print("    - layernorm (few params, little impact)")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  A initialized with Gaussian, B initialized to 0:")
print()
print("  At training start:")
print("    A ~ N(0, σ²)        (Gaussian, non-zero values)")
print("    B = 0                (all zeros)")
print("    BA = 0 * A = 0       (zero matrix)")
print()
print("  Effect:")
print("    Output = Wx + (BA)x = Wx + 0 = Wx")
print("    → At training start, LoRA output is 0")
print("    → Model behavior identical to base")
print()
print("  Why this way?")
print("    1. Training stability:")
print("      LoRA initialization doesn't suddenly change output")
print("      Loss doesn't spike")
print()
print("    2. Fair comparison:")
print("      All LoRA models behave identically before training")
print("      Performance differences purely from training")
print()
print("    3. Optimization friendly:")
print("      Gradients grow from 0")
print("      Optimizer has a stable starting point")
print()
print("  During training:")
print("    B gradually becomes non-zero")
print("    LoRA starts taking effect")
print("    Slowly deviates from base model")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  65B model QLoRA memory estimate:")
print()
print("  Model params: 65B × 0.5 bytes (4-bit) = 32.5 GB")
print("  Gradients:       0 (base frozen)")
print("  Optimizer:       0 (base frozen)")
print("  LoRA training:   50M × 8 bytes = 0.4 GB")
print("  Activations:     ~10 GB (depends on batch)")
print("  Framework overhead: ~2 GB")
print()
total = 32.5 + 0.4 + 10 + 2
print(f"  Total: {total:.1f} GB")
print()
print("  → Single A100 (80GB) can fine-tune a 65B model!")
print("  → This is QLoRA's revolutionary significance")
print()
print("  Comparison:")
print("  Full FT FP16: 65B × 2 bytes = 130 GB + optimizer etc. = 780 GB")
print("  LoRA:         130 GB + LoRA")
print("  QLoRA:        48 GB (40-50 GB)")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Merging methods:")
print()
print("  1. peft built-in (recommended):")
print("    merged_model = peft_model.merge_and_unload()")
print("    → One line of code")
print()
print("  2. Manual merge:")
print("    delta_w = (lora_B @ lora_A) * scaling")
print("    original_weight += delta_w")
print("    lora_A.zero_()")
print("    lora_B.zero_()")
print("    → No peft dependency")
print()
print("  Can it be unmerged after merging?")
print("    Theoretically: Cannot precisely unmerge")
print("    Because W' = W + BA, but after BA = 0, cannot recover")
print()
print("  In practice:")
print("    After merging, W' is the new 'base'")
print("    If you want to swap LoRA, start from original W")
print("    → Recommend keeping the original base, don't discard!")
print()
print("  At inference:")
print("    Merged: Regular model, no LoRA overhead")
print("    Unmerged: Each inference computes BA once more, slightly slower")
print()
print("  Best practice:")
print("    Train → Evaluate → Confirm effectiveness → Merge → Deploy")
print("    Keep peft version for continued training")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
