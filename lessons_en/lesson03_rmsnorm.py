"""
Lesson 3: RMSNorm — Why Do We Need Normalization?
==================================================

The "training challenge" of deep networks:
  As network depth increases, activation distributions "drift"
  - Some layers explode (very large values, e.g. 1e6)
  - Some layers vanish (very small values, e.g. 1e-6)
  - Leading to unstable gradients, training failure

Normalization layers' role:
  "Pull back" each layer's output to a reasonable range (like standardization)
  Solve Internal Covariate Shift
  Make training deep networks possible

Analogy:
  Imagine a relay race where each runner goes faster or slower
  Normalization = "recalibrate" speed after each segment
  Ensures everyone runs at a similar pace, relay stays stable

LayerNorm vs BatchNorm vs RMSNorm:

  BatchNorm:  Normalize across batch dimension, not suitable for variable-length sequences
  LayerNorm:  Normalize all features of a single sample, mainstream approach
  RMSNorm:    Simplified LayerNorm, faster computation, similar effectiveness

  Formula comparison:
    LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    RMSNorm:    x / sqrt(mean(x^2) + eps) * gamma

  RMSNorm advantages:
    + No need to subtract mean
    + No need to add bias
    + ~30% less computation
    + Similar effectiveness to LayerNorm

Run: python lessons_en/lesson03_rmsnorm.py
"""

import torch
import torch.nn as nn
import math


# ============================================================
# Part 1: Why Normalization is Needed
# ============================================================
print("=" * 60)
print("Part 1: Why Normalization is Needed")
print("=" * 60)


print("""
Problem demo: Assume a 50-layer network, each layer does y = W*x

  If W elements = 0.99:
    After 50 layers: y = 0.99^50 * x ~ 0.61 * x   (signal decay)
    Gradient also decays 50 times, approaching 0
    -> Vanishing gradient!

  If W elements = 1.01:
    After 50 layers: y = 1.01^50 * x ~ 1.64 * x   (signal amplification)
    Explodes after 50 layers
    -> Exploding gradient!

  In practice, W is learned, some layers may be too large, some too small
  -> Unstable training

Core benefits of normalization:
  + Stabilize output distribution at each layer
  + Keep gradient magnitudes moderate
  + Accelerate training convergence
  + Allow larger learning rates
""")


# Demo numerical explosion/vanishing
print("\n[Demo] Numerical problems in deep networks")
print("-" * 60)
print(f"{'Depth':<8}{'W=0.99':<15}{'W=1.01':<15}{'W=1.5':<15}{'W=2.0':<15}")
print("-" * 60)
for depth in [10, 30, 50, 100]:
    v_099 = 0.99 ** depth
    v_101 = 1.01 ** depth
    v_15 = 1.5 ** depth
    v_20 = 2.0 ** depth
    print(f"{depth:<8}{v_099:<15.4f}{v_101:<15.4f}{v_15:<15.2e}{v_20:<15.2e}")

print("\nObservation: W=1.5 reaches 6e8 after 50 layers - numerical explosion!")


# ============================================================
# Part 2: LayerNorm Implementation
# ============================================================
print("\n" + "=" * 60)
print("Part 2: LayerNorm Principle and Implementation")
print("=" * 60)


class LayerNorm(nn.Module):
    """LayerNorm - normalize across feature dimension of a single sample"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))   # scale
        self.beta = nn.Parameter(torch.zeros(dim))   # shift

    def forward(self, x):
        # x: [..., dim]
        mean = x.mean(dim=-1, keepdim=True)          # mean of last dimension
        var = x.var(dim=-1, keepdim=True)            # variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# Demo
ln = LayerNorm(8)
x = torch.randn(2, 4, 8) * 5 + 10  # arbitrary distribution
y = ln(x)

print(f"\nInput x stats: mean={x.mean():.2f}, std={x.std():.2f}")
print(f"Output y stats: mean={y.mean():.2f}, std={y.std():.2f}")
print(f"\nOutput shape: {y.shape}")
print(f"gamma shape: {ln.gamma.shape}, beta shape: {ln.beta.shape}")


# ============================================================
# Part 3: RMSNorm Implementation
# ============================================================
print("\n" + "=" * 60)
print("Part 3: RMSNorm Principle and Implementation")
print("=" * 60)


class RMSNorm(nn.Module):
    """RMSNorm - uses root mean square only, no mean subtraction"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x: [..., dim]
        # Root mean square: sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.gamma * self._norm(x)


# Demo
rms = RMSNorm(8)
y = rms(x)
print(f"\nInput x stats: mean={x.mean():.2f}, std={x.std():.2f}")
print(f"Output y stats: mean={y.mean():.2f}, std={y.std():.2f}")
print(f"\nObservation: y range is much smaller than x, but not strictly zero-mean (RMSNorm property)")


# ============================================================
# Part 4: RMSNorm vs LayerNorm Comparison
# ============================================================
print("\n" + "=" * 60)
print("Part 4: RMSNorm vs LayerNorm Comparison")
print("=" * 60)

# Performance comparison
import time

dim = 4096
batch = 32
seq = 2048
x = torch.randn(batch, seq, dim)

# Warmup
for _ in range(5):
    _ = LayerNorm(dim)(x)
    _ = RMSNorm(dim)(x)

# LayerNorm timing
start = time.time()
for _ in range(50):
    _ = LayerNorm(dim)(x)
ln_time = (time.time() - start) / 50 * 1000

# RMSNorm timing
start = time.time()
for _ in range(50):
    _ = RMSNorm(dim)(x)
rms_time = (time.time() - start) / 50 * 1000

print(f"\nPerformance comparison (dim={dim}, batch={batch}, seq={seq}):")
print(f"  LayerNorm: {ln_time:.3f} ms/call")
print(f"  RMSNorm:   {rms_time:.3f} ms/call")
print(f"  Speedup:   {ln_time/rms_time:.2f}x")

# Computation comparison
print("""
\nComputation analysis:
  LayerNorm: compute mean + var + subtract mean + divide + multiply gamma + add beta
  RMSNorm:   compute mean(x^2) + divide + multiply gamma

  RMSNorm skips:
    - One subtraction (subtract mean)
    - One addition (add bias)
    - One square root
  -> ~30-40% less computation
""")


# ============================================================
# Part 5: Where to Place Normalization
# ============================================================
print("\n" + "=" * 60)
print("Part 5: Where to Place Normalization?")
print("=" * 60)

print("""
There are 2 main positions for normalization:

1) Post-LN (Original Transformer)
   x -> Attention -> Add -> LN -> FFN -> Add -> LN
   
   Problem: Residual stream activations grow larger and larger
   Modern models rarely use this

2) Pre-LN (Modern mainstream, used by MiniMind)
   x -> LN -> Attention -> Add -> LN -> FFN -> Add
   
   Advantage: Stable residual stream, easier training
   Disadvantage: Attention layer activations may be inconsistent

Visualization:
  Post-LN:
    [x] -> [Attn] -> [+] -> [LN] -> [FFN] -> [+] -> [LN] -> output
              |                |
              +---residual-----+
    
  Pre-LN (recommended):
    [x] -> [LN] -> [Attn] -> [+] -> [LN] -> [FFN] -> [+] -> output
            |                |        |
            +---residual-----+        |
                     +---residual-----+
""")


# ============================================================
# Part 6: RMSNorm in MiniMind
# ============================================================
print("\n" + "=" * 60)
print("Part 6: RMSNorm in MiniMind")
print("=" * 60)

print("""
MiniMind uses RMSNorm in 3 places:

1) Input layer:
   x = RMSNorm(embed(x))    # Normalize right after embedding

2) Each Transformer Block (2 times):
   x = x + Attn(RMSNorm(x))     # Before attention
   x = x + FFN(RMSNorm(x))      # Before FFN

3) Final output:
   x = RMSNorm(x)            # Normalize once more before LM Head

Parameter impact:
   RMSNorm per layer: d gamma parameters
   MiniMind 8 layers, hidden=512: 8 x 512 = 4096 parameters
   Proportion: < 0.1% (negligible overhead)

Implementation (PyTorch):
```python
class MiniMindRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
```
""")


# ============================================================
# Part 7: Normalization Variants Summary
# ============================================================
print("\n" + "=" * 60)
print("Part 7: Normalization Variants Summary")
print("=" * 60)

comparison = """
+------------+----------+----------+----------+----------+
| Method     | Norm Dim | Compute  | Use Case | Model    |
+------------+----------+----------+----------+----------+
| BatchNorm  | batch    | Medium   | CNN      | ResNet   |
| LayerNorm  | sample   | Medium   | General  | BERT     |
| RMSNorm    | sample   | Low      | LLM      | LLaMA    |
| DeepNorm   | sample   | Medium   | Deep     | GPT-3    |
| GroupNorm  | channels | Medium   | Small b  | ConvNeXt |
+------------+----------+----------+----------+----------+

Why do LLMs all use RMSNorm?
  - LLM sequence lengths vary greatly, BatchNorm is unusable
  - Faster than LayerNorm with similar effectiveness
  - More stable during training (smaller numerical range)
"""

print(comparison)


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Numerical Explosion Analysis
  Given W = 1.5 matrix multiplication, after how many layers does output exceed 1e6?

[Exercise 2] LayerNorm Output Range
  After LayerNorm, what are the approximate mean and std?

[Exercise 3] RMSNorm Mean
  RMSNorm doesn't make data strictly zero-mean, why?

[Exercise 4] Pre-LN vs Post-LN
  Why do modern LLMs (LLaMA, GPT) all use Pre-LN?

[Exercise 5] Gamma Parameter
  RMSNorm has a learnable parameter gamma (default initialized to 1)
  Does gamma change after training? What values does it take?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  1.5^n = 1e6")
print("  n * log(1.5) = log(1e6)")
print("  n = 6 / log10(1.5) = 6 / 0.176 = 34.1")
print("  -> Approximately 35 layers before numerical explosion to 1e6")

# Verification
import math
print("\n  Verification:")
for n in [10, 20, 30, 35, 40]:
    val = 1.5 ** n
    print(f"    1.5^{n} = {val:.2e}")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  After LayerNorm: mean ~ 0, std ~ 1")
print("  With unchanged gamma and beta, output strictly mean=0, std=1")
print("  After training, gamma and beta adjust the output distribution")

# Verification
ln = LayerNorm(8)
x = torch.randn(100, 8) * 10 + 5
y = ln(x)
print(f"\n  Verification: y mean={y.mean():.4f}, y std={y.std():.4f}")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  RMSNorm formula: x / sqrt(mean(x^2) + eps) * gamma")
print("  No mean subtraction step, so output is not necessarily zero-mean")
print("  Example: input [1, 2, 3, 4]")
print("  RMS = sqrt(mean([1,4,9,16])) = sqrt(7.5) ~ 2.739")
print("  After scaling ~ [0.365, 0.730, 1.095, 1.461]")
print("  Mean ~ 0.913, not 0")

# Demo
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
rms = RMSNorm(4)
y = rms(x)
print(f"\n  Verification: input {x.tolist()[0]}")
print(f"        output {[round(v, 3) for v in y.tolist()[0]]}")
print(f"        mean = {y.mean():.3f}")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  Pre-LN advantages:")
print("  1. Residual stream variance stays stable across layers (no exponential growth)")
print("  2. Gradients propagate directly through residual stream during training, bypassing LN")
print("  3. Easier to train very deep (100+) models")
print("  Post-LN disadvantages:")
print("  1. Activations on the residual path grow exponentially with depth")
print("  2. Requires warmup learning rate, unstable training")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Yes, it changes! Gamma is initialized to 1, then adjusts during training")
print("  Gamma values reflect the importance of each dimension")
print("  - Important dimensions: larger gamma (amplified)")
print("  - Unimportant dimensions: smaller gamma (diminished)")
print("  - Normalizable dimensions: gamma ~ 0")
print("  After training, most gamma values are in the 0.5-2.0 range")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Why Normalization is Needed
   - Deep networks have numerical explosion/vanishing problems
   - Normalization stabilizes training
   - Allows larger learning rates

2. LayerNorm
   - Formula: (x - mean) / sqrt(var) * gamma + beta
   - Normalizes across feature dimension of a single sample
   - General-purpose solution

3. RMSNorm (used by MiniMind)
   - Formula: x / sqrt(mean(x^2)) * gamma
   - 30%+ simpler than LayerNorm
   - Similar effectiveness, faster speed
   - The choice for modern LLMs

4. Position Selection
   - Pre-LN (LN before sublayer) is mainstream
   - More stable training than Post-LN
   - MiniMind also uses Pre-LN

5. Simple Implementation
   - Only 1 parameter (gamma)
   - Negligible parameter increase
   - Minimal inference overhead

Next: Lesson 4 - RoPE (How does the model know word positions?)
""")
