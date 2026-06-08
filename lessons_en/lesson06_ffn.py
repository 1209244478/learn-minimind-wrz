"""
Lesson 6: FFN — Feed-Forward Network: Memory Refinement After Attention
=======================================================================

Attention lets words exchange information, but each position's expressive
power on its own is limited. The role of FFN (Feed-Forward Network):
independently perform "memory refinement" on each position.

Core structure (SwiGLU variant, used by MiniMind):
  FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

Breakdown:
  1. gate_proj: maps hidden_size-dimensional vector to a higher dimension (intermediate_size)
  2. SiLU activation: gate_proj output passes through activation function (gating signal)
  3. up_proj: another projection to higher dimension (information path)
  4. Element-wise multiplication: gating * information = selective retention
  5. down_proj: maps back to hidden_size dimensions

Intuitive understanding:
  Attention = "communication" between words
  FFN = each word's own "thinking"

  After Attention collects contextual information, FFN processes each
  position independently — refining and storing important information,
  filtering out noise.

Run: python lessons_en/lesson06_ffn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Step 1: The Simplest FFN
# ============================================================

print("=" * 60)
print("Experiment 1: The Simplest FFN — Two Linear Transformations")
print("=" * 60)

class SimpleFFN(nn.Module):
    """Simplest FFN: Linear -> ReLU -> Linear"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x)
        x = self.down_proj(x)
        return x

hidden_size = 64
intermediate_size = 256

ffn = SimpleFFN(hidden_size, intermediate_size)
x = torch.randn(1, 8, hidden_size)
out = ffn(x)

print(f"Input shape: {x.shape}")
print(f"After up-projection: [{x.shape[0]}, {x.shape[1]}, {intermediate_size}]")
print(f"Output shape: {out.shape}")
print(f"\nFFN dimension changes: {hidden_size} -> {intermediate_size} -> {hidden_size}")
print(f"First up-project (expand information), then down-project (compress and refine)")


# ============================================================
# Step 2: The Role of Activation Functions
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: Why Do We Need Activation Functions?")
print("=" * 60)

x = torch.linspace(-3, 3, 100)

# ReLU: max(0, x)
relu_out = F.relu(x)

# GELU: x * Phi(x), smoother than ReLU
gelu_out = F.gelu(x)

# SiLU (Swish): x * sigmoid(x), used by MiniMind
silu_out = F.silu(x)

print("Activation Function Comparison:")
print(f"  ReLU: max(0, x) — simple and crude, negatives go directly to zero")
print(f"  GELU: x * Phi(x) — smooth version of ReLU, negatives not fully zeroed")
print(f"  SiLU: x * sigma(x) — self-gated, x multiplied by its own probability")
print(f"\n  x=-2: ReLU={F.relu(torch.tensor(-2.0)):.4f}, "
      f"GELU={F.gelu(torch.tensor(-2.0)):.4f}, "
      f"SiLU={F.silu(torch.tensor(-2.0)):.4f}")
print(f"  x=0:  ReLU={F.relu(torch.tensor(0.0)):.4f}, "
      f"GELU={F.gelu(torch.tensor(0.0)):.4f}, "
      f"SiLU={F.silu(torch.tensor(0.0)):.4f}")
print(f"  x=2:  ReLU={F.relu(torch.tensor(2.0)):.4f}, "
      f"GELU={F.gelu(torch.tensor(2.0)):.4f}, "
      f"SiLU={F.silu(torch.tensor(2.0)):.4f}")

# What happens without activation functions?
linear_no_act = nn.Sequential(
    nn.Linear(hidden_size, intermediate_size, bias=False),
    nn.Linear(intermediate_size, hidden_size, bias=False),
)
linear_with_act = SimpleFFN(hidden_size, intermediate_size)

x_test = torch.randn(1, 4, hidden_size)
out_no_act = linear_no_act(x_test)
out_with_act = linear_with_act(x_test)

print(f"\nWithout activation: two linear layers = one linear layer (equivalent to hidden_size -> hidden_size)")
print(f"With activation: introduces non-linearity, expressive power far exceeds a single linear layer")


# ============================================================
# Step 3: SwiGLU — The FFN Variant Used by MiniMind
# ============================================================

print("\n" + "=" * 60)
print("Experiment 3: SwiGLU — Gated Linear Unit")
print("=" * 60)

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (used by MiniMind)

    FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    Difference from simple FFN:
      - Added a gate_proj (gating path)
      - SiLU(gate) * up implements selective information transfer
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))   # Gating signal: decides how much info to retain
        up = self.up_proj(x)                # Information path: content to be passed
        return self.down_proj(gate * up)    # gating * information -> selective retention


ffn_simple = SimpleFFN(hidden_size, intermediate_size)
ffn_swiglu = SwiGLUFFN(hidden_size, intermediate_size)

x = torch.randn(1, 8, hidden_size)
out_simple = ffn_simple(x)
out_swiglu = ffn_swiglu(x)

simple_params = sum(p.numel() for p in ffn_simple.parameters())
swiglu_params = sum(p.numel() for p in ffn_swiglu.parameters())

print(f"Simple FFN parameter count: {simple_params:,}")
print(f"SwiGLU FFN parameter count: {swiglu_params:,}")
print(f"SwiGLU adds gate_proj, parameter count is about {swiglu_params/simple_params:.1f}x")

# Visualize the gating mechanism
torch.manual_seed(42)
x_demo = torch.randn(1, 1, hidden_size)
gate = F.silu(ffn_swiglu.gate_proj(x_demo))
up = ffn_swiglu.up_proj(x_demo)

print(f"\nGating signal (gate_proj -> SiLU):")
print(f"  Range: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
print(f"  Mean: {gate.mean().item():.4f}")
print(f"  SiLU output near 0 -> this dimension is 'closed'")
print(f"  SiLU output large -> this dimension is 'open'")

gated = gate * up
print(f"\nAfter gating (gate * up):")
print(f"  Range: [{gated.min().item():.4f}, {gated.max().item():.4f}]")
print(f"  Gating achieves selective information transfer!")


# ============================================================
# Step 4: FFN Parameter Count Analysis
# ============================================================

print("\n" + "=" * 60)
print("Experiment 4: FFN Accounts for the Bulk of Model Parameters")
print("=" * 60)

configs = [
    ("MiniMind-Small", 768, 2048),
    ("MiniMind-Medium", 1024, 2816),
    ("LLaMA-7B", 4096, 11008),
]

for name, hidden, intermediate in configs:
    gate_params = hidden * intermediate
    up_params = hidden * intermediate
    down_params = intermediate * hidden
    total_ffn = gate_params + up_params + down_params
    total_attn = 4 * hidden * hidden  # Q, K, V, O projections (simplified)

    print(f"\n{name}: hidden={hidden}, intermediate={intermediate}")
    print(f"  gate_proj: {hidden}x{intermediate} = {gate_params:,}")
    print(f"  up_proj:   {hidden}x{intermediate} = {up_params:,}")
    print(f"  down_proj: {intermediate}x{hidden} = {down_params:,}")
    print(f"  FFN total params: {total_ffn:,}")
    print(f"  Attention total params (simplified): {total_attn:,}")
    print(f"  FFN/Attention ratio: {total_ffn/total_attn:.1f}x")

print("\n-> FFN parameters are typically 2-3x that of Attention!")
print("-> intermediate_size is about 2.7-3x hidden_size")


# ============================================================
# Step 5: FFN's Per-Position Independence
# ============================================================

print("\n" + "=" * 60)
print("Experiment 5: FFN Operates Independently on Each Position")
print("=" * 60)

ffn = SwiGLUFFN(64, 256)
x = torch.randn(1, 4, 64)

# Method 1: Batch input
out_batch = ffn(x)

# Method 2: Per-position input
out_positions = []
for i in range(4):
    pos_out = ffn(x[:, i:i+1, :])
    out_positions.append(pos_out)
out_manual = torch.cat(out_positions, dim=1)

print(f"Batch input output: {out_batch.shape}")
print(f"Per-position input output: {out_manual.shape}")
print(f"Both methods produce identical results: {torch.allclose(out_batch, out_manual, atol=1e-5)}")
print("\n-> FFN operates on each position independently, no cross-position information exchange")
print("-> This complements Attention: Attention communicates, FFN thinks")


# ============================================================
# Step 6: FFN as "Key-Value Memory"
# ============================================================

print("\n" + "=" * 60)
print("Experiment 6: FFN's Memory Perspective")
print("=" * 60)

print("""
Research shows that FFN can be viewed as a "key-value memory system":

  Each row of up_proj = a "key" (key pattern)
  Each column of down_proj = a "value" (value pattern)

  FFN(x) = sum( activation(key_i . x) * value_i )

  That is: input x computes match scores (activation values) with all keys,
  then weighted sum of corresponding values based on match scores.

  [Mathematical Breakdown of FFN as Key-Value Memory]

    FFN(x) = W_down · σ(W_up · x)

    Step 1: W_up · x  →  each row of W_up is a "key"
      h_i = key_i · x    (how well does input x match pattern key_i?)

    Step 2: σ(h_i)  →  activation (how strongly should we use this knowledge?)
      a_i = σ(h_i)       (0 = not relevant, 1 = very relevant)

    Step 3: W_down · a  →  each column of W_down is a "value"
      y = Σ a_i · value_i  (weighted sum of knowledge)

    Library Analogy:
      key_i = a book's index/title (what topic does this book cover?)
      value_i = the book's content (what knowledge does it contain?)
      h_i = how well the input matches this book's topic
      a_i = should we read this book? (0 = skip, 1 = read carefully)
      y = combined knowledge from all relevant books

  Analogy:
    key = "this is a subject position" -> value = "features for filling in nouns"
    key = "this is a negation position" -> value = "features for reversing sentiment"

  Attention is memory lookup between "word and word"
  FFN is memory lookup between "input and knowledge"
""")


# ============================================================
# Step 7: FFN Configuration in MiniMind
# ============================================================

print("=" * 60)
print("Experiment 7: FFN Configuration in MiniMind")
print("=" * 60)

# MiniMind actual configuration
print("MiniMind's FFN configuration:")
print("  hidden_size = 768")
print("  intermediate_size = 2048 (about 2.67x hidden_size)")
print("  Activation function = SiLU (SwiGLU variant)")
print("  No bias (bias=False)")
print()
print("FFN parameter count per Transformer Block:")
hidden = 768
inter = 2048
params = 3 * hidden * inter  # gate + up + down
print(f"  3 x {hidden} x {inter} = {params:,} = {params/1e6:.2f}M")
print(f"  (gate_proj + up_proj + down_proj)")


# ============================================================
# Deep Dive: The Essence of FFN
# ============================================================
print("\n" + "=" * 60)
print("Deep Dive: The Essence and Analogy of FFN")
print("=" * 60)

print("""
[Analogy: FFN as "Knowledge Base Retrieval"]
------------------------------------------
FFN can be viewed as a giant "key-value memory":

  down_proj weights W_down: [hidden x intermediate]
  -> Each row = a "knowledge entry"
  -> Intermediate dimension = number of knowledge entries

  Flow:
    1. gate_proj decides "what knowledge I care about" (query)
    2. up_proj maps input to "knowledge space"
    3. SiLU(gate) * up = "filter knowledge by relevance"
    4. down_proj converts "activated knowledge" back to hidden representation

  Library analogy:
    Assume each row of down_proj is a book's content
    Input x decides "which books I want to read"
    Gate decides "the relevance of these books"
    Output = mixture of relevant books' content


[Diagram: SwiGLU FFN Data Flow]
-------------------------------

       x (hidden_size)
       |
       +--------------+
       |              |
       v              v
   +--------+    +--------+
   |gate_proj|   |up_proj  |
   +---+----+   +----+---+
       |             |
       v             v
      SiLU           x
       |             |
       +-----O-------+  <- element-wise multiplication (gating)
             |
             v
        +--------+
        |down_proj|
        +---+----+
            |
            v
       output (hidden_size)


[Why Up-Project?]
-----------------
  Analogy: when solving problems, first diverge (consider multiple possibilities)
       then converge (select the most relevant)

  Mathematically:
    - Low-dimensional space: limited expressiveness, hard to separate different concepts
    - High-dimensional space: strong expressiveness, can learn complex mappings
    - Intermediate dimension is typically 2.67-4x hidden
    - MiniMind uses 4x (e.g., hidden=512, inter=2048)

  Empirical values: larger models prefer larger intermediate dimensions
    GPT-3: 4x hidden
    LLaMA: (8/3)x hidden (SwiGLU)
    PaLM:  4x hidden


[Activation Function Comparison]
-------------------------------
  ReLU:     f(x) = max(0, x)
            Simple, but negatives go directly to zero (dead neurons)

  GELU:     f(x) = x * Phi(x) (Gaussian)
            Smooth, more stable training, used by BERT

  SiLU:     f(x) = x * sigmoid(x) (Swish)
            Used by LLaMA / MiniMind
            Smoother than ReLU, slightly faster than GELU
            Slightly better than GELU on large models

  SwiGLU:   f(x, gate) = SiLU(gate) * x
            Adds gating on top of SiLU
            The de facto standard for current large models
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Necessity of Activation Functions
  What happens if FFN has no activation function (only linear transformations)?

[Exercise 2] FFN Parameter Count
  With hidden=512, inter=2048, what is the parameter count of SwiGLU?
  How many times the hidden size is this equivalent to?

[Exercise 3] FFN vs Attention Parameter Count
  Given hidden=512, num_heads=8, head_dim=64
  Single-layer Attention parameter count = ?
  Single-layer FFN parameter count = ?
  How many times is FFN compared to Attention?

[Exercise 4] Gating Mechanism
  SwiGLU adds a gate_proj compared to a regular FFN
  What is the purpose of the extra parameters?

[Exercise 5] FFN Thinking vs Attention Communicating
  What is the division of labor between Attention and FFN in the model?
  What happens if FFN is removed?

[Exercise 6] Why Multiply down_proj by 2/3?
  The LLaMA paper says SwiGLU's intermediate dimension should be (2/3) * 4 * hidden
  Why multiply by 2/3?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Without activation functions, multiple linear layers = one linear layer")
print("  Mathematical proof:")
print("    y = W2(W1*x + b1) + b2 = W2*W1*x + (W2*b1 + b2)")
print("    Let W = W2*W1, b = W2*b1 + b2")
print("    y = W*x + b, equivalent to a single linear layer")
print()
print("  Deep models degenerate into shallow models, unable to learn complex relationships")
print("  -> Activation functions are the 'soul' of deep learning")

# Demo
import torch
import torch.nn as nn
linear = nn.Sequential(
    nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8),
    nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 4)
)
total = sum(p.numel() for p in linear.parameters())
print(f"  6-layer pure linear network parameter count: {total}")

# Equivalent
single = nn.Linear(8, 4)
print(f"  Equivalent single-layer parameter count: {sum(p.numel() for p in single.parameters())}")
print(f"  6 layers are useless, equivalent to 1 layer!")

# Exercise 2
print("\n[Exercise 2 Answer]")
hidden = 512
inter = 2048
gate = hidden * inter
up = hidden * inter
down = inter * hidden
total = gate + up + down
print(f"  gate_proj: {hidden} x {inter} = {gate:,}")
print(f"  up_proj:   {hidden} x {inter} = {up:,}")
print(f"  down_proj: {inter} x {hidden} = {down:,}")
print(f"  Total: {total:,} = {total/1e6:.2f}M")
print(f"  Equivalent to {total/(hidden*hidden):.1f}x hidden^2 = {(total/3)/(hidden*hidden):.1f}x single Linear layer")

# Exercise 3
print("\n[Exercise 3 Answer]")
hidden = 512
num_heads = 8
head_dim = 64
inter = 2048

# Attention: Q, K, V, O four projections
attn_params = 4 * hidden * hidden
print(f"  Attention: 4 x {hidden}^2 = {attn_params:,} = {attn_params/1e6:.2f}M")

# FFN: gate, up, down three projections (SwiGLU)
ffn_params = 3 * hidden * inter
print(f"  FFN (SwiGLU): 3 x {hidden} x {inter} = {ffn_params:,} = {ffn_params/1e6:.2f}M")

print(f"  FFN / Attention = {ffn_params/attn_params:.2f}x")
print(f"  -> FFN parameters are ~3x that of Attention, the bulk of model parameters")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  The extra gate_proj serves as 'selective activation'")
print()
print("  Regular FFN:  y = activation(W1*x) * W2")
print("             ^ All dimensions are activated")
print()
print("  SwiGLU:    y = (SiLU(W_gate*x) * W_up*x) * W_down")
print("             ^ Dimensions where SiLU output is near 0, the entire channel is 'closed'")
print("             ^ Dimensions near 1, information passes through completely")
print()
print("  Effects of gating:")
print("    - Forces FFN to learn 'which dimensions to activate'")
print("    - Reduces useless computation (filtered out by gating)")
print("    - Improves model expressiveness (sparse activation)")
print("    - Empirically SwiGLU > GELU > ReLU")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Attention's role: information communication")
print("    - Lets different positions exchange information")
print("    - Each position's output = weighted average of all positions")
print("    - Complexity: O(N^2) (N is sequence length)")
print()
print("  FFN's role: information processing")
print("    - Processes each position independently (no cross-position communication)")
print("    - Extracts, transforms, and memorizes features")
print("    - Complexity: O(N) (once per position)")
print()
print("  If FFN is removed:")
print("    - Model only has 'information mixing', no 'information processing'")
print("    - Performance drops significantly (~50% loss)")
print("    - Like having only a library, but no brain")
print()
print("  Both are indispensable, together forming the Transformer Block")
print("  -> Attention makes information flow, FFN makes information 'think'")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  Regular FFN intermediate dimension = 4 * hidden")
print("  SwiGLU adds a gate_proj, increasing total parameter count")
print()
print("  To keep total parameter count unchanged, adjust intermediate dimension:")
print("    Regular:  2 * hidden * inter = 2 * hidden * (4*hidden) = 8 * hidden^2")
print("    SwiGLU: 3 * hidden * inter (gate + up + down)")
print("    Let 3 * hidden * inter = 8 * hidden^2")
print("    inter = 8/3 * hidden")
print()
print("  Paper (Shazeer 2020) found through experiments:")
print("    When inter = 2/3 * 4 * hidden = 8/3 * hidden")
print("    SwiGLU performs best at the same parameter count")
print()
print("  -> This is an engineering 'parameter budget balancing' technique")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
print("""
1. FFN processes each position independently: up-project -> activate -> down-project
2. Activation functions introduce non-linearity; without them, two linear layers = one
3. SwiGLU = SiLU(gate) * up, gating mechanism selectively retains information
4. FFN parameters are 2-3x that of Attention, the bulk of the model
5. FFN can be viewed as "key-value memory": keys match input, values provide knowledge
6. Attention communicates + FFN thinks = the two pillars of Transformer

Data flow so far:
  ... -> Attention -> Residual Connection -> RMSNorm -> FFN -> Residual Connection -> ...
       Word communication    Add original info   Normalize   Position processing  Add original info

Next -> lesson07_block.py: Assemble Attention + FFN into a Transformer Block
""")
