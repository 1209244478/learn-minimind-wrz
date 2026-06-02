"""
Lesson 4: RoPE — How Does the Model Know Word Positions?
=========================================================

"Cat eats fish" and "Fish eats cat" contain the same words, but mean completely different things.
The model must know each word's position to understand word order.

Naive idea: Give each position a number (0, 1, 2, ...), add it to the vector.
Problem: Numbers are absolute values, the model can't easily understand relative relationships like "distance between position 1 and 3 = 2".

RoPE (Rotary Position Embedding) approach:
  Instead of adding position info, encode position through "rotation"!
  Treat vectors as points on a 2D plane, rotate position m's vector by m*theta angle.

Key properties:
  1. Dot product depends only on relative position difference (m-n), naturally encoding relative position
  2. Long-range decay: larger position difference -> smaller dot product
  3. No extra parameters needed, pure mathematical operations

Run: python lessons_en/lesson04_rope.py
"""

import torch
import math


# ============================================================
# Step 1: Why Position Encoding is Needed
# ============================================================

print("=" * 60)
print("Experiment 1: Without position info, the model can't distinguish word order")
print("=" * 60)

word_cat = torch.tensor([1.0, 0.5])
word_eat = torch.tensor([0.3, 0.8])
word_fish = torch.tensor([0.7, 0.2])

sentence1 = [word_cat, word_eat, word_fish]   # Cat eats fish
sentence2 = [word_fish, word_eat, word_cat]   # Fish eats cat

print("Cat eats fish:", [w.tolist() for w in sentence1])
print("Fish eats cat:", [w.tolist() for w in sentence2])
print("Same set of words, but completely different meanings!")
print("-> The model needs to know each word's position")


# ============================================================
# Step 2: 2D Rotation — Core Intuition of RoPE
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: 2D Rotation Encodes Position")
print("=" * 60)

def rotate_2d(vec, angle):
    """2D rotation: rotate vector by angle radians"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y = vec[0], vec[1]
    return torch.tensor([x * cos_a - y * sin_a, x * sin_a + y * cos_a])

theta = 0.5  # base angle

q_pos0 = rotate_2d(word_cat, 0 * theta)
q_pos1 = rotate_2d(word_cat, 1 * theta)
q_pos2 = rotate_2d(word_cat, 2 * theta)

print(f"Original vector:     {word_cat.tolist()}")
print(f"Position 0 (0*theta):   {q_pos0.tolist()}")
print(f"Position 1 (1*theta):   {q_pos1.tolist()}")
print(f"Position 2 (2*theta):   {q_pos2.tolist()}")

dot_01 = torch.dot(q_pos0, q_pos1)
dot_12 = torch.dot(q_pos1, q_pos2)
print(f"\nDot product pos 0&1: {dot_01:.4f}")
print(f"Dot product pos 1&2: {dot_12:.4f}")
print(f"Similar (relative distance is 1 for both): {abs(dot_01 - dot_12) < 0.01}")

dot_02 = torch.dot(q_pos0, q_pos2)
print(f"Dot product pos 0&2: {dot_02:.4f} (distance 2, smaller dot product)")
print("-> Greater distance -> smaller dot product (long-range decay)")


# ============================================================
# Step 3: Extending to Higher Dimensions — Grouped Rotation
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: Different Dimensions Use Different Frequencies")
print("=" * 60)

dim = 8
base = 10000.0

freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
print(f"Dimension: {dim}, split into {dim//2} groups")
print(f"Group frequencies theta: {freqs.tolist()}")
print(f"Group 0 has highest frequency ({freqs[0]:.4f}) -> captures local position")
print(f"Group {dim//2-1} has lowest frequency ({freqs[-1]:.6f}) -> captures global position")

print("\nPosition |  Group 0 angle  |  Group 3 angle")
print("-" * 40)
for pos in [0, 1, 5, 10, 50, 100]:
    angle_0 = pos * freqs[0].item()
    angle_3 = pos * freqs[3].item()
    print(f"  {pos:3d}  |  {angle_0:8.4f}   |  {angle_3:10.6f}")

print("\n-> High-freq group: large angle difference between pos 1 and 2, can distinguish adjacent positions")
print("-> Low-freq group: small angle difference between pos 1 and 2, but noticeable between pos 1 and 100")


# ============================================================
# Step 4: Manual RoPE Implementation
# ============================================================

def rope_precompute(dim, max_seq_len, base=10000.0):
    """Precompute RoPE cos and sin tables"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)
    cos_table = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin_table = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return cos_table, sin_table


def apply_rope(x, cos, sin):
    """Apply rotary position encoding to input vectors"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_h = cos[..., :half]
    sin_h = sin[..., :half]
    out = torch.cat([x1 * cos_h - x2 * sin_h,
                     x2 * cos_h + x1 * sin_h], dim=-1)
    return out


print("\n" + "=" * 60)
print("Experiment 4: Manual RoPE Implementation")
print("=" * 60)

head_dim = 8
seq_len = 5
num_heads = 2
batch_size = 1

torch.manual_seed(42)
xq = torch.randn(batch_size, seq_len, num_heads, head_dim)

cos_table, sin_table = rope_precompute(head_dim, max_seq_len=10)

print(f"Input shape: {xq.shape}")
print(f"cos table shape: {cos_table.shape}")
print(f"sin table shape: {sin_table.shape}")

cos = cos_table[:seq_len]
sin = sin_table[:seq_len]

xq_rope = apply_rope(xq, cos, sin)

print(f"\nPosition 0 original vector (head0): {xq[0, 0, 0].tolist()}")
print(f"Position 0 rotated vector (head0):  {xq_rope[0, 0, 0].tolist()}")
print(f"Position 1 original vector (head0): {xq[0, 1, 0].tolist()}")
print(f"Position 1 rotated vector (head0):  {xq_rope[0, 1, 0].tolist()}")

print(f"\nPosition 0 cos values: {cos[0].tolist()}")
print(f"Position 0 sin values: {sin[0].tolist()}")
print("Position 0: all cos=1, sin=0 -> no rotation at position 0!")


# ============================================================
# Step 5: Verify RoPE's Relative Position Property
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5: RoPE Encodes Relative Position")
print("=" * 60)

head_dim = 16
seq_len = 20

torch.manual_seed(0)
q_vec = torch.randn(1, 1, 1, head_dim)
k_vec = torch.randn(1, 1, 1, head_dim)

cos_t, sin_t = rope_precompute(head_dim, max_seq_len=100)

print("Relative distance | Dot product")
print("-" * 30)
for m in [5]:
    q_rot = apply_rope(q_vec, cos_t[m:m+1], sin_t[m:m+1])
    for n_offset in [0, 1, 2, 3, 5, 10, 20, 50]:
        n = m + n_offset
        k_rot = apply_rope(k_vec, cos_t[n:n+1], sin_t[n:n+1])
        dot = torch.dot(q_rot[0, 0, 0], k_rot[0, 0, 0]).item()
        print(f"   {n_offset:2d}       |  {dot:.4f}")

print("\n-> Larger relative distance -> smaller dot product (long-range decay)")
print("-> This is RoPE's core advantage: attention naturally favors nearby positions")


# ============================================================
# Step 6: Comparison with Other Position Encodings
# ============================================================

print("\n" + "=" * 60)
print("Experiment 6: Position Encoding Methods Compared")
print("=" * 60)

methods = {
    "Absolute Position (Original Transformer)": "Learn a vector for each position, add to input\n  Con: Can't extrapolate to unseen lengths",
    "ALiBi": "Add linear bias to attention scores\n  Pro: Naturally supports length extrapolation\n  Con: No absolute position info",
    "RoPE (used by MiniMind)": "Encode position through rotation\n  Pro: Encodes relative position + long-range decay + no extra params\n  Con: Length extrapolation needs extra tricks (e.g. YaRN)",
}

for name, desc in methods.items():
    print(f"\n{name}:")
    print(f"  {desc}")


# ============================================================
# Step 7: RoPE in MiniMind
# ============================================================

print("\n" + "=" * 60)
print("Experiment 7: RoPE Implementation in MiniMind")
print("=" * 60)

def precompute_freqs_cis(dim, end=32768, rope_base=1e6):
    """MiniMind's RoPE precomputation (simplified, without YaRN scaling)"""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """MiniMind's RoPE application function"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    cos_h, sin_h = cos[..., :half], sin[..., :half]
    q_out = torch.cat([q1 * cos_h - q2 * sin_h, q2 * cos_h + q1 * sin_h], dim=-1)
    k_out = torch.cat([k1 * cos_h - k2 * sin_h, k2 * cos_h + k1 * sin_h], dim=-1)
    return q_out, k_out

head_dim = 64
num_heads = 8
num_kv_heads = 4
seq_len = 16
batch_size = 2
hidden_size = num_heads * head_dim

freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=2048)
print(f"Precomputed cos table shape: {freqs_cos.shape}")
print(f"Precomputed sin table shape: {freqs_sin.shape}")
print(f"Computed once, shared for all positions!")

xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
xk = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)

cos = freqs_cos[:seq_len]
sin = freqs_sin[:seq_len]

xq_rope, xk_rope = apply_rotary_pos_emb(xq, xk, cos, sin)

print(f"\nQ shape: {xq.shape} -> after rotation: {xq_rope.shape}")
print(f"K shape: {xk.shape} -> after rotation: {xk_rope.shape}")
print(f"RoPE doesn't change vector shape, only direction!")

norm_before = xq[0, 0, 0].norm().item()
norm_after = xq_rope[0, 0, 0].norm().item()
print(f"\nVector norm before rotation: {norm_before:.4f}")
print(f"Vector norm after rotation:  {norm_after:.4f}")
print(f"Norm unchanged: {abs(norm_before - norm_after) < 0.01}")
print("-> Rotation only changes direction, not magnitude")


# ============================================================
# Step 8: RoPE's base Parameter and Frequency Range
# ============================================================

print("\n" + "=" * 60)
print("Experiment 8: Impact of base Parameter")
print("=" * 60)

for base in [100.0, 10000.0, 1000000.0]:
    freqs = 1.0 / (base ** (torch.arange(0, 16, 2).float() / 16))
    print(f"\nbase = {base:.0f}:")
    print(f"  Highest frequency: {freqs[0]:.4f} (fastest rotation, distinguishes adjacent positions)")
    print(f"  Lowest frequency:  {freqs[-1]:.6f} (slowest rotation, distinguishes distant positions)")
    period = 2 * math.pi / freqs[-1].item()
    print(f"  Lowest freq period: {period:.0f} positions")

print("\nMiniMind uses base = 1,000,000 (100x larger than default 10,000)")
print("-> Larger base -> lower minimum frequency -> longer period")
print("-> Suitable for longer sequences (supports farther position discrimination)")


# ============================================================
# Step 9: RoPE's Complete Flow in Attention
# ============================================================

print("\n" + "=" * 60)
print("Experiment 9: RoPE's Complete Flow in Attention")
print("=" * 60)

print("""
RoPE usage flow in Attention:

1. Input x: [batch, seq_len, hidden_size]
2. Linear projection to get Q, K, V:
   xq = q_proj(x)  -> [batch, seq_len, num_heads, head_dim]
   xk = k_proj(x)  -> [batch, seq_len, num_kv_heads, head_dim]
   xv = v_proj(x)  -> [batch, seq_len, num_kv_heads, head_dim]

3. Apply RoPE to Q and K (NOT V!):
   xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

4. Compute attention scores:
   scores = xq @ xk^T / sqrt(head_dim)
   -> Thanks to RoPE, scores automatically include relative position info!

5. Weighted sum with attention:
   output = softmax(scores) @ xv

Key points:
  - Only rotate Q and K, NOT V
  - cos/sin are precomputed at model initialization, no learning needed
  - RoPE adds zero trainable parameters
""")


# ============================================================
# Step 10: RoPE Intuition Diagram
# ============================================================
print("\n" + "=" * 60)
print("Experiment 10: Rotation Intuition")
print("=" * 60)

print("""
Imagine you're at the center of a clock, treating each word's vector as a hand:

       0 deg
       |
   ----+----
   |       |
180deg *   0deg
   |       |
   ----+----
       |
      90deg

Position 0: hand points in some direction (e.g. 12 o'clock = 0 deg)
Position 1: hand rotates clockwise by theta
Position 2: hand rotates clockwise by 2*theta
Position m: hand rotates clockwise by m*theta

Different dimensions' "hands" rotate at different speeds:
  Dim 0-1:  fast rotation (high freq) -> distinguishes adjacent words
  Dim 2-3:  slow rotation (low freq) -> distinguishes paragraph-level positions

-> High-freq dimensions handle "nearby relationships" (local)
-> Low-freq dimensions handle "distant relationships" (global)

Like satellite navigation:
  Uses both GPS (high freq, precise) and compass (low freq, general direction)
  Both combined for accurate positioning
""")


# ============================================================
# Step 11: RoPE's Mathematical Properties
# ============================================================
print("\n" + "=" * 60)
print("Experiment 11: Three Key Mathematical Properties of RoPE")
print("=" * 60)

print("""
Property 1: Relative Position Encoding
---------------------------------------
  <R(m*theta)*q, R(n*theta)*k> = <q, k> * cos((m-n)*theta)

  Meaning: Dot product of two rotated vectors depends only on relative distance (m-n)
  Example: q at position 5, k at position 7
      Distance = 2
      Same result as q at position 10, k at position 12
      -> Model naturally understands the concept of "distance 2"

Property 2: Long-Range Decay
-----------------------------
  When |m-n| is large, cos((m-n)*theta) approaches 0 (or oscillates)
  -> Greater distance -> lower attention score
  -> Model automatically focuses more on nearby positions

  Note: At very large distances, cos oscillates (between +1 and -1)
  This is why RoPE is hard to extrapolate to very long sequences
  (Can be addressed with YaRN / NTK-aware interpolation)

Property 3: Length Independence
-------------------------------
  cos/sin tables only depend on head_dim
  Trained with length 2048, can infer with length 4096
  As long as cos/sin tables can index the corresponding positions
  -> Naturally supports "train short, infer long"
""")


# ============================================================
# Step 12: RoPE vs Traditional Position Encodings
# ============================================================
print("\n" + "=" * 60)
print("Experiment 12: RoPE vs Traditional Position Encodings")
print("=" * 60)

print("""
+--------------------+--------------+--------------+--------------+
| Method             | Position Info| Relative Pos | Extrapolation|
+--------------------+--------------+--------------+--------------+
| Absolute (BERT)    | + Added      | - Hard       | - Cannot     |
| Relative Bias (T5) | - No absolute| + To scores  | + Can        |
| ALiBi              | - No absolute| + Linear bias| + Can        |
| RoPE               | + Implicit   | + In dot prod| ~ Needs trick|
+--------------------+--------------+--------------+--------------+

RoPE's uniqueness:
  - Has absolute position (rotation m*theta contains m's info)
  - Has relative position (dot product depends only on m-n)
  - Has long-range decay (cos property)
  -> "Three in one", very elegant!
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Why doesn't V need rotation?
  RoPE only applies rotation to Q and K, but not V. Why?

[Exercise 2] Relative Position Encoding
  Suppose q is at position 3, k is at position 7
  What is the relative distance? Is it the same as q at position 10, k at position 14?

[Exercise 3] base Parameter Selection
  Larger base -> lower minimum frequency -> longer period.
  If training sequence length is 2048 and you want to extrapolate to 16384, should base be larger or smaller?

[Exercise 4] Dimension Pairing
  RoPE splits a d-dim vector into d/2 groups of 2 dims each
  Why 2 dims per group, not 4 or 8?

[Exercise 5] RoPE and KV Cache
  During KV Cache inference, each new token is at position m
  Q needs rotation at position m, K is already cached
  At which position's rotation was K cached?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  V represents 'what information is at this position', doesn't need position info")
print("  Q and K compute 'which positions are relevant', need position info")
print("  Attention formula: softmax(QK^T)V")
print("  Q and K determine weights, V provides content")
print("  -> V is not rotated, content semantics unchanged, only attention distribution changes")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  q at position 3, k at position 7: relative distance = 7 - 3 = 4")
print("  q at position 10, k at position 14: relative distance = 14 - 10 = 4")
print("  -> Same relative distance, same dot product!")
print("  This is RoPE's relative position property: <R(m*theta)q, R(n*theta)k> depends only on m-n")

# Demo
head_dim = 16
cos_t, sin_t = rope_precompute(head_dim, max_seq_len=100)
torch.manual_seed(42)
q = torch.randn(1, 1, 1, head_dim)
k = torch.randn(1, 1, 1, head_dim)
for m, n in [(3, 7), (10, 14), (0, 4)]:
    q_rot = apply_rope(q, cos_t[m:m+1], sin_t[m:m+1])
    k_rot = apply_rope(k, cos_t[n:n+1], sin_t[n:n+1])
    dot = torch.dot(q_rot[0, 0, 0], k_rot[0, 0, 0]).item()
    print(f"  q@{m}, k@{n}: distance={n-m}, dot={dot:.4f}")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Should increase base")
print("  Larger base -> lower minimum frequency -> longer period")
print("  Longer period means positions can be distinguished at greater distances")
print("  MiniMind uses base=1e6 (vs default 1e4) for this reason")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  2D rotation is the simplest complete rotation (SO(2) group)")
print("  In 2D, rotation has a clean closed form: cos/sin matrix")
print("  Higher-dimensional rotations (4D, 8D) are much more complex")
print("  2D pairing also ensures the dot product property: <R(m)*q, R(n)*k> = f(m-n)")
print("  This property doesn't hold for arbitrary dimensional rotations")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  K was cached with the rotation of its original position")
print("  When generating token at position m:")
print("    - Q is computed fresh and rotated at position m")
print("    - K was computed and rotated at position n when that token was generated")
print("    - The cached K already has position n's rotation applied")
print("  -> No need to re-rotate K during inference, just use the cached value")
print("  This is why KV Cache works with RoPE!")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Why Position Encoding
   - Without it, the model can't distinguish word order
   - "Cat eats fish" vs "Fish eats cat" need different representations

2. RoPE Core Idea
   - Encode position through vector rotation
   - Position m rotates by m*theta angle
   - Dot product naturally encodes relative position

3. Multi-frequency Design
   - Different dimension groups use different rotation frequencies
   - High freq: local position differences
   - Low freq: global position information

4. Key Properties
   - Relative position: dot product depends only on m-n
   - Long-range decay: farther positions get less attention
   - No extra parameters: pure math operations

5. MiniMind's RoPE
   - base = 1,000,000 (supports longer sequences)
   - Precomputed cos/sin tables
   - Applied to Q and K, not V

Next: Lesson 5 - Attention (How do words attend to each other?)
""")
