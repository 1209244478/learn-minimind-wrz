"""
Lesson 16: YaRN — Teaching Models to Generalize Sequence Length
================================================================

Problem: A model trained on 2048-length sequences — can it work on 8192?
  Standard RoPE: No! Performance drops dramatically
  Reason: The model only saw small positions during training,
          but encounters large positions during inference

What is "Length Extrapolation"?
  Training sequence length: 2048
  Inference sequence length: 8192 (4x beyond training length)

  Training:  The model has seen positions 1-2048
  Inference: Position 5000 suddenly appears — the model never learned it

YaRN (Yet another RoPE extensioN) solution:
  1. Scale positions: pos' = pos / s
  2. Correct attention scores: compensate for entropy changes
  3. Scale most frequencies proportionally
  4. Works even when training is short and inference is long

Run: python lessons_en/lesson16_yarn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: The Length Extrapolation Problem
# ============================================================

print("=" * 60)
print("Part 1: The Length Extrapolation Problem")
print("=" * 60)

print("""
Suppose a model is trained on 2048-length sequences:

During training:
  - Position 1 has word: "today"
  - Position 100 has word: "weather"
  - Position 500 has word: "nice"
  - Position 2000 has word: "right"

During inference, feeding a 4096-length text:
  - Positions 1-2048: Model has seen these, performs normally
  - Positions 2049-4096: Model has never seen these!
  - Result: Gibberish, performance collapse

Why does this happen?
  Reason 1: Position encodings exceed the training range
  Reason 2: Attention distribution is completely different from training
  Reason 3: Large-valued sine/cosine values that the model never learned
""")

def rope_freqs(head_dim, position, base=10000):
    """Compute RoPE frequencies"""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = position * freqs
    return torch.cos(angles), torch.sin(angles)

print("\n[Demo] RoPE trigonometric values at different positions")
print("-" * 60)
print("Position   | cos(θ) (low-freq)   | cos(θ) (high-freq)")
print("-" * 60)
head_dim = 64
for pos in [100, 1000, 2048, 4096, 8192]:
    cos_vals, _ = rope_freqs(head_dim, torch.tensor([pos]))
    print(f"Position {pos:5d} | {cos_vals[0].item():+.4f}            | {cos_vals[-1].item():+.4f}")

print("\nObservation: High-frequency components vary drastically at different positions")
print("During training, positions < 2048, the model only learned small-range variation patterns")


# ============================================================
# Part 2: Position Interpolation (PI)
# ============================================================

print("\n" + "=" * 60)
print("Part 2: Position Interpolation (PI)")
print("=" * 60)

print("""
Simplest approach: "Compress" all positions into the training range

Original:   position = 1, 2, ..., 4096 (extrapolation)
Interp.:    position' = position / 4    (scale by 4x)
            = 0.25, 0.5, ..., 1024     (within training range)

Formula:  pos' = pos × (L_train / L_inference)

Pros:
  + Simple, one line of code
  + Preserves position information learned during training

Cons:
  - Position "resolution" decreases, distant positions become less distinguishable
  - Still needs fine-tuning to recover performance
""")

print("\n[Demo] Position Interpolation (scale_factor=4)")
print("-" * 60)
print("Original Pos | Scaled Pos  | Mapping Range")
print("-" * 60)
scale = 4
train_len = 2048
for pos in [100, 1000, 2048, 4096, 8192]:
    scaled = pos / scale
    print(f"{pos:8d}    | {scaled:8.1f}    | {'Within training' if scaled <= train_len else 'Beyond training'}")


# ============================================================
# Part 3: YaRN — Fine-grained Position Extrapolation
# ============================================================

print("\n" + "=" * 60)
print("Part 3: YaRN — Fine-grained Position Extrapolation")
print("=" * 60)

print("""
YaRN's core insight:
  RoPE frequencies range from high to low
  - Low-frequency dimensions: Encode coarse positions (large range)
  - High-frequency dimensions: Encode fine positions (small range)

  Simple scaling hurts high-frequency dimensions (loses detail)
  Simple scaling works well for low-frequency dimensions

YaRN's approach:
  1. Divide RoPE frequencies into 3 groups:
     - High-frequency: No scaling (preserve detail)
     - Mid-frequency: Smooth transition
     - Low-frequency: Full scaling

  2. For each dimension d, compute:
     h(d) = scaling factor (dimension-dependent)

  3. Scale position: pos' = pos × h(d)

  4. Attention temperature correction:
     t' = 0.1 × ln(s) + 1    (s is the scale factor)
     attn' = attn / t'

Effects:
  + Better than simple interpolation
  + No fine-tuning needed (or very little)
  + Works for 4x-16x length extrapolation
""")


# ============================================================
# Part 4: Complete YaRN Implementation
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Complete YaRN Implementation")
print("=" * 60)


def get_yarn_mscale(scale_factor):
    """Compute YaRN's mscale (temperature correction)"""
    if scale_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scale_factor) + 1.0


def precompute_yarn_freqs_cis(dim, end, original_max=2048, scale_factor=4.0,
                               beta_fast=32, beta_slow=1):
    """YaRN frequency precomputation"""
    freqs_base = 1.0 / (10000 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if scale_factor <= 1.0:
        t = torch.arange(end)
        freqs = torch.outer(t, freqs_base).float()
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        return freqs_cos, freqs_sin

    inv_freq_extrapolation = 1.0 / (scale_factor * 10000 ** (
        torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    inv_freq_interpolation = 1.0 / (10000 ** (
        torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    low_freq_factor = scale_factor
    high_freq_factor = 1.0
    freq_wavelen = 2 * math.pi / inv_freq_interpolation

    low_freq_wavelen = original_max / low_freq_factor
    high_freq_wavelen = original_max / high_freq_factor

    smooth = (freq_wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    smooth = smooth.clamp(0, 1)
    inv_freq = (1 - smooth) * inv_freq_extrapolation + smooth * inv_freq_interpolation

    t = torch.arange(end)
    freqs = torch.outer(t, inv_freq).float()

    mscale = get_yarn_mscale(scale_factor)

    freqs_cos = torch.cat([torch.cos(freqs) * mscale, torch.cos(freqs) * mscale], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs) * mscale, torch.sin(freqs) * mscale], dim=-1)
    return freqs_cos, freqs_sin


print("\n[Demo] YaRN frequencies vs Original RoPE frequencies")
print("-" * 60)
print("Frequency values at position 5000:")
print(f"{'Dimension':<10}{'Original RoPE':<20}{'YaRN (4x)':<20}")
print("-" * 60)
cos_orig, _ = precompute_yarn_freqs_cis(32, 5001, scale_factor=1.0)
cos_yarn, _ = precompute_yarn_freqs_cis(32, 5001, scale_factor=4.0)

for d in [0, 8, 16, 24]:
    orig_val = cos_orig[5000, d].item()
    yarn_val = cos_yarn[5000, d].item()
    print(f"{d:<10}{orig_val:<20.4f}{yarn_val:<20.4f}")


# ============================================================
# Part 5: Applying YaRN
# ============================================================

print("\n" + "=" * 60)
print("Part 5: Applying YaRN to Attention")
print("=" * 60)

print("""
Using YaRN only requires replacing the RoPE frequency computation:

```python
# Original RoPE
freqs_cos, freqs_sin = precompute_freqs_cis(dim, end)

# YaRN
freqs_cos, freqs_sin = precompute_yarn_freqs_cis(
    dim, end,
    original_max=2048,    # Max length during training
    scale_factor=8.0,     # Scale factor (8192/2048=4, but 8-16 is common)
)
```

The attention computation stays the same — only the position encoding changes.
""")

print("\n[Experiment] Verifying YaRN-supported length extrapolation")
print("-" * 60)
print(f"Training length: 2048, Target length: 8192, Scale factor: 4x")

mscale = get_yarn_mscale(4.0)
print(f"\nYaRN mscale = {mscale:.4f}")
print(f"Meaning: Attention scores need to be divided by {mscale:.4f} for temperature correction")
print(f"         (Reduces sharpness of attention distribution, helps model attend to longer ranges)")


# ============================================================
# Part 6: Comparison and Selection
# ============================================================

print("\n" + "=" * 60)
print("Part 6: Length Extrapolation Method Comparison")
print("=" * 60)

comparison = """
+--------------+----------+----------+------------------+
| Method       | Difficulty | Quality  | Fine-tuning Needed|
+--------------+----------+----------+------------------+
| None (direct)| 0        | Poor     | 0                |
| PI           | 1        | Fair     | 100% length FT   |
| NTK-aware    | 3        | Good     | 10% length FT    |
| YaRN         | 5        | Excellent| 1% length FT     |
| ABF          | 5        | Excellent| 10% length FT    |
+--------------+----------+----------+------------------+

Recommendations:
  - Simple needs: Linear interpolation + full-length fine-tuning
  - Balanced:     YaRN + 1% fine-tuning
  - Maximum:      YaRN + key-sample fine-tuning
"""

print(comparison)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Length Extrapolation Problem
   - Model trained on 2048-length, inference at 4096 causes collapse
   - Root cause: Large positions never seen during training

2. Position Interpolation (PI)
   - Compress all positions into the training range
   - Simple to implement, but needs fine-tuning

3. YaRN Improvements
   - Independently decide scaling degree for each dimension
   - Low-frequency dimensions: large scaling; High-frequency: no scaling
   - Temperature correction: mscale = 0.1*ln(s) + 1
   - Dramatically better results, almost no fine-tuning needed

4. Practical Application
   - In MiniMind, replace via precompute_yarn_freqs_cis
   - Supports training at 2K, extending to 8K/16K inference

Next: Lesson 17 - mHC (Manifold-Constrained Hyper-Connections)
""")


# ============================================================
# Deep Understanding: YaRN's "Frequency-Band Speed Control"
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: YaRN's 'Frequency-Band Speed Control'")
print("=" * 60)

print("""
[Analogy: YaRN = A Telescope's Zoom Lens]
------------------------------------------

  A trained telescope (LLaMA, 2K context):
    Designed for close-up viewing (short context)
    Focus distance is fixed

  Want to see far (long context):
    Simple stretching: Blurry
    Replace the lens: Expensive and troublesome

  YaRN:
    Smart zoom: Different adjustments for different focal segments
    - Low frequency (near): Almost no adjustment
    - High frequency (far): Major adjustment
    -> Results close to replacing the lens, at minimal cost


[Why Transformers Can't Directly Extrapolate]
---------------------------------------------

  RoPE principle review:
    Frequency θ_i = base^{-2i/d}
    Smaller i → larger θ (high frequency, short period)
    Larger i  → smaller θ (low frequency, long period)

  The problem:
    During training, θ_i is in range [θ_min, θ_max]
    During inference, position m exceeds the max training position
    -> m * θ_i becomes very large
    -> Rotation angle "overflows", losing discriminability

  Analogy:
    A clock hand rotates 6 degrees per second
    Watching 60 seconds (1 revolution) is fine
    Watching 3600 seconds (60 revolutions) — can't tell the position
    -> Because the period is too short, too many revolutions


[Three Length Extrapolation Methods Compared]
---------------------------------------------

  +----------+----------------+------------------+----------+
  | Method   | Core Idea      | Fine-tuning Need | Quality  |
  +----------+----------------+------------------+----------+
  | PI       | Linear adjust  | Yes (long text)  | Medium   |
  | NTK-aware| Adjust base    | No               | Medium   |
  | YaRN     | Per-band adjust| No / very little | High     |
  +----------+----------------+------------------+----------+

  PI (Position Interpolation):
    Scale position m: m' = m * (L_train / L_target)
    -> Makes inference positions fall within training range
    -> Simple but loses high-frequency information

  NTK-aware:
    Adjust the base frequency, making low frequencies slower
    -> Don't scale positions, change frequencies instead
    -> High frequencies unchanged, low frequencies denser

  YaRN (this lesson):
    Combines PI + NTK + attention scaling
    -> Best results


[YaRN's Core Formula]
---------------------

  For each frequency dimension i, introduce a scaling function r(i):

  h_θ(m, i) = m * r(i) * θ_i

  Where r(i) is the key:
    +--------------------------------------------+
    | r(i) =                                     |
    |   1                    if λ(i) < α        | <- Low freq, unchanged
    |   (λ(i) - α) / (β - α) if α ≤ λ(i) ≤ β  | <- Mid freq, linear interp
    |   1 / s                if λ(i) > β        | <- High freq, large scaling
    +--------------------------------------------+

  Where:
    λ(i) = wavelength = 2π / θ_i
    s = L_target / L_train (scaling ratio)
    α, β: thresholds (empirical values α=1, β=32)

  Analogy:
    Low-frequency dimensions (large i): Slow rotation, long period
      -> Stretching positions won't "overflow"
      -> No adjustment needed
    High-frequency dimensions (small i): Fast rotation, short period
      -> Easy to "overflow"
      -> Must scale significantly


[YaRN's Three Components]
-------------------------

  1. NTK-by-parts (frequency scaling)
    Different dimensions use different scaling
    -> More refined than uniform scaling

  2. Attention Scaling
    For long contexts, the "temperature" of attention scores must be adjusted

    Formula: attn / sqrt(t)
    t = 0.1 * ln(s) + 1     (YaRN empirical formula)

    The larger s (longer extrapolation), the larger t
    -> Softmax becomes "smoother"
    -> Avoids over-focusing on specific positions

  3. Minimal fine-tuning (optional)
    Fine-tune with 0.1% long texts
    -> Lets the model adapt to the new distribution
    -> Significantly improves results


[YaRN Inference Code (Simplified)]
----------------------------------

  def precompute_yarn_freqs_cis(
      dim, max_seq_len, base=10000,
      scale_factor=8.0,
      alpha=1, beta=32
  ):
      freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

      # 1. Compute wavelength for each dimension
      wavelengths = 2 * pi / freqs

      # 2. Segmented scaling
      ramp = (wavelengths - alpha) / (beta - alpha)
      ramp = ramp.clamp(0, 1)
      r_inter = (1 - ramp) + ramp / scale_factor
      r_intra = 1 / scale_factor

      # 3. Apply scaling
      freqs = freqs * r_inter
      # High freq uses r_intra, low freq uses r_inter

      # 4. Attention scaling
      attn_factor = 0.1 * math.log(scale_factor) + 1.0

      return freqs, attn_factor


[YaRN vs Other Methods Performance]
------------------------------------

  Experiment (LLaMA-7B -> 16K context):

  +--------------+----------+--------------+
  | Method       | Perplexity| Retrieval Acc|
  +--------------+----------+--------------+
  | No extrap.   | > 100    | Random       |
  | PI           | 6.5      | 65%          |
  | NTK-aware    | 5.8      | 75%          |
  | YaRN         | 4.2      | 92%          |
  | YaRN + FT    | 3.9      | 95%          |
  +--------------+----------+--------------+

  -> YaRN nearly reaches the training limit
  -> Retrieval tasks benefit especially


[Practical Application Choices]
-------------------------------

  1. Model trained at 4K, want to use 16K at inference:
    -> YaRN (scale_factor=4)

  2. Model trained at 2K, want to use 32K at inference:
    -> YaRN (scale_factor=16)
    -> With a small amount of fine-tuning

  3. Model trained at 8K, want to use 128K at inference:
    -> YaRN + ReRoPE
    -> May need retraining

  4. Uncertain:
    -> Try YaRN first, add fine-tuning if needed
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Length Extrapolation Problem
  A model trained on 4K context — what happens at 16K inference? Why?

[Exercise 2] PI Scale Factor
  Trained on 4K, want to extrapolate to 16K. What should s be?

[Exercise 3] High-frequency vs Low-frequency
  In RoPE, which dimension needs scaling more: high-freq or low-freq? Why?

[Exercise 4] YaRN's α and β
  What do α and β control? What happens if you increase/decrease them?

[Exercise 5] Attention Scaling
  Why do we need to scale attention scores for long contexts?
  What happens without scaling?

[Exercise 6] YaRN's Limitations
  Can YaRN extend a 4K model to 1M context? What are the constraints?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

print("\n[Exercise 1 Answer]")
print("  Symptoms:")
print("    - Model outputs gibberish / repetitions")
print("    - Perplexity explodes (from 5 to 100+)")
print("    - Attention scores become abnormal (overly dispersed or concentrated)")
print()
print("  Root cause:")
print("    RoPE encoding: q_i * exp(i * m * θ)")
print("    During training: m ∈ [0, 4096]")
print("    During inference: m ∈ [0, 16384]")
print()
print("    Rotation angle = m * θ_i")
print("    m increases 4x, angle increases 4x")
print("    High-freq dimensions (large θ): Angle overflows, loses discrimination")
print("    Low-freq dimensions (small θ): Angle remains reasonable")
print()
print("  Analogy:")
print("    Clock second hand (high freq): Too many revolutions to distinguish")
print("    Clock hour hand (low freq): A few revolutions still distinguishable")

print("\n[Exercise 2 Answer]")
print("  s = L_target / L_train = 16384 / 4096 = 4")
print()
print("  PI method:")
print("    Position scaling: m' = m / s = m / 4")
print("    -> Maps [0, 16384] to [0, 4096]")
print("    -> Falls within training range")
print()
print("  YaRN method:")
print("    s=4 is used as the scaling factor for high-frequency dimensions")
print("    Low-frequency dimensions scale less (or not at all)")
print()
print("  In practice:")
print("    4x extrapolation is relatively easy, YaRN works well")
print("    16x extrapolation is more extreme, needs fine-tuning")
print("    64x+ is essentially impossible")

print("\n[Exercise 3 Answer]")
print("  High-frequency dimensions need scaling more")
print()
print("  Frequency analysis:")
print("    High freq (small i, large θ): Short period")
print("    Low freq (large i, small θ): Long period")
print()
print("  Example (LLaMA, dim=128, base=10000):")
print("    i=0:   θ = 1.0,       period = 2π/1.0 ≈ 6.28")
print("    i=32:  θ ≈ 0.01,      period = 2π/0.01 ≈ 628")
print("    i=63:  θ ≈ 0.000103,  period ≈ 61000")
print()
print("  The problem:")
print("    High freq i=0, period ≈ 6")
print("    Position m=4096, rotates 4096/6 ≈ 683 revolutions")
print("    Position m=16384, rotates 16384/6 ≈ 2730 revolutions")
print("    -> 4x more revolutions, angles become indistinguishable")
print()
print("    Low freq i=32, period ≈ 628")
print("    Position m=4096, rotates 4096/628 ≈ 6.5 revolutions")
print("    Position m=16384, rotates 16384/628 ≈ 26 revolutions")
print("    -> Still in reasonable range, angles are distinguishable")
print()
print("  YaRN's handling:")
print("    High freq (small i): Large scaling (1/s)")
print("    Low freq (large i): No scaling (1)")

print("\n[Exercise 4 Answer]")
print("  α and β are the piecewise thresholds for YaRN's scaling function")
print()
print("  α (lower bound):")
print("    Dimensions with wavelength < α: No scaling (r=1)")
print("    Default α=1, short wavelength (high frequency)")
print("    -> These dimensions change rapidly, scaling would destroy detail")
print()
print("  β (upper bound):")
print("    Dimensions with wavelength > β: Full scaling (r=1/s)")
print("    Default β=32, long wavelength (low frequency)")
print("    -> These dimensions change slowly, safe to scale")
print()
print("  Between α and β:")
print("    Smooth linear interpolation")
print()
print("  Adjusting α and β:")
print("    Increase α: More dimensions get no scaling (preserves more detail)")
print("    Decrease α: More dimensions get scaled (better extrapolation)")
print("    Increase β: More dimensions get full scaling (stronger extrapolation)")
print("    Decrease β: Fewer dimensions get full scaling (more conservative)")

print("\n[Exercise 5 Answer]")
print("  Why scale attention scores for long contexts:")
print()
print("  Problem:")
print("    After length extrapolation, attention scores become more 'peaked'")
print("    -> Model over-focuses on nearby positions")
print("    -> Ignores distant but important information")
print()
print("  Without scaling:")
print("    Attention distribution becomes too sharp")
print("    -> Long-range dependencies are lost")
print("    -> Model behaves like it has very short effective context")
print()
print("  With YaRN temperature correction:")
print("    t = 0.1 * ln(s) + 1")
print("    attn' = attn / t")
print("    -> Softmax becomes smoother")
print("    -> Model can attend to longer ranges")

print("\n[Exercise 6 Answer]")
print("  YaRN cannot extend a 4K model to 1M context directly")
print()
print("  Limitations:")
print("    1. Scale factor too large:")
print("       1M / 4K = 256x extrapolation")
print("       Even with frequency scaling, position resolution becomes too low")
print()
print("    2. Attention pattern mismatch:")
print("       The model has never seen 1M-token attention patterns")
print("       No amount of position scaling can fix this")
print()
print("    3. KV Cache memory:")
print("       1M context needs enormous KV Cache")
print("       Even with quantization, memory is prohibitive")
print()
print("  Practical limits:")
print("    4x-8x extrapolation: Reliable with YaRN")
print("    8x-16x: Possible with YaRN + fine-tuning")
print("    16x-32x: Difficult, needs specialized techniques")
print("    32x+: Essentially requires retraining on longer sequences")
