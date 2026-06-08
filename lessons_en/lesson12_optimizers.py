"""
Lesson 12: Optimizers — The Soul That Determines How Models Learn
================================================================

Training a model involves two key questions:
  1. How does the model "compute" predictions?  → Forward pass
  2. How does the model "correct" errors?       → Optimizer

The role of an Optimizer:
  Update model parameters based on gradients computed from the loss function

Why do we need different optimizers?
  Simple gradient descent has many problems:
  - Slow, prone to oscillation
  - Easily stuck in local optima
  - Different parameters need different learning rates

This lesson evolves step by step:
  SGD → Momentum → Adam → AdamW → Muon

Run: python lessons_en/lesson12_optimizers.py
"""

import torch
import torch.nn as nn
import math


# ============================================================
# Part 1: What is Optimization?
# ============================================================

print("=" * 60)
print("Part 1: What is Optimization?")
print("=" * 60)

print("""
Imagine you're descending a mountain at night, only feeling the slope beneath your feet (gradient).
Your goal: Reach the lowest point in the valley (minimum loss).

Optimizer = Your "walking strategy".

  - How fast to walk?        → Learning rate
  - Use momentum?            → Momentum
  - Can it be adaptive?      → Adaptive learning rate (Adam)
""")

# Simple demo: Finding the minimum of y = x^2
def f(x):
    return x ** 2

def grad_f(x):
    return 2 * x

# Different optimizers finding the minimum
_state = {}

def optimize(optimizer_name, x_start=5.0, lr=0.1, steps=20):
    x = torch.tensor(x_start, requires_grad=True)
    history = [x.item()]

    if optimizer_name == "Momentum":
        _state["v"] = torch.zeros_like(x)

    for _ in range(steps):
        loss = f(x)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        # Update based on different optimizers
        with torch.no_grad():
            if optimizer_name == "SGD":
                x -= lr * x.grad
            elif optimizer_name == "Momentum":
                # Simplified Momentum
                beta = 0.9
                _state["v"] = beta * _state["v"] + x.grad
                x -= lr * _state["v"]

        history.append(x.item())
    return history


# Training process visualization
print("\n[Experiment 1] Different optimizers finding minimum of y=x^2:")
print("-" * 60)
print(f"{'Step':<6}{'SGD':<20}{'Momentum':<20}")
print("-" * 60)

# Reset Momentum state
optimize.v = None
hist_sgd = optimize("SGD", x_start=5.0, lr=0.1, steps=15)
hist_mom = optimize("Momentum", x_start=5.0, lr=0.1, steps=15)

for i in range(0, 16, 2):
    print(f"{i:<6}{hist_sgd[i]:<20.4f}{hist_mom[i]:<20.4f}")

print("\nObservation: Momentum converges faster and more stably than SGD!")


# ============================================================
# Part 2: SGD — The Simplest Optimizer
# ============================================================

print("\n" + "=" * 60)
print("Part 2: SGD — Stochastic Gradient Descent")
print("=" * 60)

print("""
SGD (Stochastic Gradient Descent) — The simplest optimizer

Core idea:
  1. Compute loss L
  2. Backpropagate to get gradient ∇L
  3. Update parameters: θ = θ - lr * ∇L

Formula:
  θ_new = θ - lr * ∇L
       = θ - learning_rate × gradient

Analogy:
  You're descending a mountain, lr determines your step size
  Step too large  → Overshoot the minimum
  Step too small  → Descent too slow
""")

class SimpleSGD:
    """Simplified implementation of SGD optimizer"""

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        """Update parameters"""
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad.data

    def zero_grad(self):
        """Clear gradients"""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# Compare: Our implementation vs PyTorch built-in
x = torch.tensor(5.0, requires_grad=True)
optimizer = SimpleSGD([x], lr=0.1)

print("\nOur SGD implementation training 10 steps:")
for i in range(10):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        print(f"  Step {i}: x = {x.item():.4f}, loss = {loss.item():.4f}")


# ============================================================
# Part 3: SGD with Momentum — Adding Inertia
# ============================================================

print("\n" + "=" * 60)
print("Part 3: SGD with Momentum — Adding Inertia")
print("=" * 60)

print("""
Problem: SGD easily oscillates on both sides of a "canyon"
       ↓
Solution: Add "inertia" (momentum)

Core idea:
  - Maintain a velocity v (momentum)
  - Gradient doesn't just update parameters, it also updates velocity
  - Velocity accumulates, forming "inertia"

Formula:
  v_new = β * v + ∇L          (β is momentum coefficient, usually 0.9)
  θ_new = θ - lr * v_new

Analogy:
  A ball rolling down a mountain
  - After accumulating speed, it can roll over small bumps
  - It won't stop immediately on flat areas
""")

class MomentumSGD:
    """SGD with Momentum"""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.velocities[i] = self.momentum * self.velocities[i] + p.grad.data
                p.data = p.data - self.lr * self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# Demo
x = torch.tensor(5.0, requires_grad=True)
optimizer = MomentumSGD([x], lr=0.1, momentum=0.9)

print("\nSGD with Momentum training 10 steps:")
for i in range(10):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        print(f"  Step {i}: x = {x.item():.4f}, loss = {loss.item():.4f}")


# ============================================================
# Part 4: Adam — Adaptive Learning Rate
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Adam — The King of Adaptive Learning Rate")
print("=" * 60)

print("""
Problem:
  - Different parameters may need different learning rates
  - Some parameters update frequently, some are sparse

Solution: Each parameter maintains its own learning rate
  - Frequently updated parameters → Decrease learning rate
  - Sparsely updated parameters → Increase learning rate

Adam (Adaptive Moment Estimation) core ideas:
  1. First moment m: Exponential moving average of gradients (similar to Momentum)
  2. Second moment v: Exponential moving average of squared gradients (learning rate adjustment)
  3. Bias correction: Corrects initial estimation inaccuracy

Formula:
  m_t = β1 * m_{t-1} + (1-β1) * ∇L
  v_t = β2 * v_{t-1} + (1-β2) * (∇L)²
  m_hat = m_t / (1 - β1^t)
  v_hat = v_t / (1 - β2^t)
  θ = θ - lr * m_hat / (√v_hat + ε)
""")

class SimpleAdam:
    """Simplified implementation of Adam optimizer"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p.data) for p in self.params]  # Second moment
        self.t = 0  # Step count

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad = p.grad.data

                # Update first and second moments
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update parameters
                p.data = p.data - self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# Demo
x = torch.tensor(5.0, requires_grad=True)
optimizer = SimpleAdam([x], lr=0.5)

print("\nAdam training 20 steps (lr=0.5):")
for i in range(20):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 4 == 0:
        print(f"  Step {i}: x = {x.item():.4f}, loss = {loss.item():.4f}")


# ============================================================
# Part 5: AdamW — Decoupled Weight Decay
# ============================================================

print("\n" + " = " * 30)
print("Part 5: AdamW — Correct Weight Decay")
print("=" * 60)

print("""
Problem: Adam's weight decay is problematic
  - L2 regularization in Adam interacts with adaptive learning rate
  - Causing weight decay to become ineffective

Solution: AdamW "decouples" weight decay
  - No longer add L2 regularization to the gradient
  - Instead, directly subtract wd * θ from parameters

Formula:
  m_t = β1 * m_{t-1} + (1-β1) * ∇L
  v_t = β2 * v_{t-1} + (1-β2) * (∇L)²
  m_hat = m_t / (1 - β1^t)
  v_hat = v_t / (1 - β2^t)
  θ = θ - lr * (m_hat / (√v_hat + ε) + wd * θ)

Advantages:
  - Better weight decay effect
  - More stable training
  - Current de facto standard for LLM training
""")

class SimpleAdamW:
    """AdamW optimizer"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # Adam step
                grad = p.grad.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Note: Weight decay is directly applied to parameters (decoupled)
                p.data = p.data - self.lr * (
                    m_hat / (v_hat.sqrt() + self.eps)
                    + self.weight_decay * p.data
                )

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ============================================================
# Part 6: Muon — Modern Optimizer
# ============================================================

print("\n" + "=" * 60)
print("Part 6: Muon — Next-Generation Optimizer")
print("=" * 60)

print("""
Muon (Momentum Orthogonalized by Newton-schulz) — A new optimizer proposed in 2024

Core idea (one-sentence version):
  Muon = Momentum + Orthogonalization
  Make update directions perpendicular to each other, training more efficient

[Intuition: What is "Orthogonalization"?]

  Imagine you're finding the lowest point in a valley (optimization):

  Regular gradient descent:
    Take a step in the steepest direction each time
    Problem: if the valley is a narrow ellipse, you oscillate back and forth
    → Too much movement north-south, too little east-west

  After orthogonalization:
    Equalize the step size in "north-south" and "east-west" directions
    → No more oscillation, head straight for the lowest point
    → This is the effect of orthogonalization!

  Mathematically:
    Gradient matrix singular values are uneven (directions are imbalanced)
    Orthogonalization = make all directions have the same "step size"
    → Each direction contributes equally, learning is more balanced

  Analogy:
    Regular optimization = walking with a big left step, small right step (wobbling)
    Orthogonalization    = equal-sized steps with both feet (walking straight)

Why orthogonalization?
  - Gradient matrix singular value distribution is uneven
  - Large singular values dominate updates → Uneven learning
  - After orthogonalization, all directions contribute equally

Formula:
  g = ∇L                      # Original gradient
  m = β * m + g              # Momentum
  g_ortho = NewtonSchulz(m)  # Newton-Schulz orthogonalization
  θ = θ - lr * g_ortho

Newton-Schulz orthogonalization:
  Use 5 matrix multiplication iterations to convert a matrix into an orthogonal matrix
  No SVD decomposition needed, fast speed
  (You don't need to understand the math details, just know it makes directions more uniform)

Why is Muon powerful?
  - Training speed 2x faster than AdamW
  - Memory savings (no second moment needed)
  - More stable convergence
  - Already adopted by Kimi and other large models
""")

def newton_schulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz orthogonalization (5 iterations)"""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


class SimpleMuon:
    """Muon optimizer - Simplified version"""

    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None or p.ndim < 2:
                # 1D parameters (norms, biases) skip Muon, use SGD
                if p.grad is not None:
                    p.data = p.data - self.lr * p.grad.data
                continue

            grad = p.grad.data

            # Momentum update
            self.velocities[i] = self.momentum * self.velocities[i] + grad

            if self.nesterov:
                grad = self.velocities[i] + self.momentum * (
                    self.velocities[i] - self.velocities[i] / self.momentum
                )
            else:
                grad = self.velocities[i]

            # Newton-Schulz orthogonalization
            grad_ortho = newton_schulz5(grad)

            # Update parameters
            p.data = p.data - self.lr * grad_ortho

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# Demo
print("\nDemo: Optimizing 2D matrix parameters")

# 2D matrix
W = torch.randn(4, 8) * 5
W.requires_grad_(True)

optimizer = SimpleMuon([W], lr=0.01)

print(f"Initial W Frobenius norm: {W.norm():.4f}")
for i in range(5):
    loss = (W ** 2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  Step {i+1}: loss = {loss.item():.4f}, ||W|| = {W.norm():.4f}")


# ============================================================
# Part 7: How to Choose in MiniMind
# ============================================================

print("\n" + "=" * 60)
print("Part 7: MiniMind's Optimizer Selection")
print("=" * 60)

print("""
Common optimizer configuration for MiniMind training:

1. 2D parameters (q_proj, k_proj, v_proj, o_proj, fc, proj) → Muon
   - Matrix parameters, orthogonalization works well
   - Fast training, memory efficient

2. 1D parameters (norms, biases) → AdamW
   - Scalar/vector, not suitable for orthogonalization
   - More stable with AdamW

3. Embedding / LM Head → AdamW
   - Special layers, handled separately

Code example (from MiniMind):
```python
# Separate parameters
muon_params = [p for n, p in model.named_parameters()
               if p.ndim >= 2 and 'embed' not in n and 'head' not in n]
adamw_params = [p for n, p in model.named_parameters()
                if p.ndim < 2 or 'embed' in n or 'head' in n]

# Create two optimizers
optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
optimizer.add_param_group({
    'params': adamw_params,
    'optimizer': AdamW(adamw_params, lr=3e-4, weight_decay=0.01)
})
```
""")


# ============================================================
# Optimizer Comparison
# ============================================================

print("=" * 60)
print("Optimizer Comparison Summary")
print("=" * 60)

comparison = """
┌──────────────┬────────────────┬─────────────┬────────────────┐
│ Optimizer    │ Core Idea      │ Pros        │ Cons           │
├──────────────┼────────────────┼─────────────┼────────────────┤
│ SGD          │ Basic gradient │ Simple,     │ Slow, prone to │
│              │ descent        │ stable      │ oscillation    │
│ Momentum     │ Add inertia    │ Faster      │ Hard to tune   │
│              │                │ convergence │ learning rate  │
│ Adam         │ Adaptive       │ Fast        │ Weight decay   │
│              │ learning rate  │ convergence │ issues         │
│ AdamW        │ Decoupled      │ General,    │ Large memory   │
│              │ weight decay   │ stable      │ usage          │
│ Muon         │ Gradient       │ Faster,     │ 2D parameters  │
│              │ orthogonalize  │ memory save │ only           │
└──────────────┴────────────────┴─────────────┴────────────────┘

Recommendations:
  - Small experiments → AdamW
  - Medium training   → Muon + AdamW mixed
  - Large model       → Muon + AdamW mixed
"""

print(comparison)


# ============================================================
# Deep Understanding: Optimizer "Superpowers"
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: Optimizer 'Superpowers'")
print("=" * 60)

print("""
[Analogy: Optimizer = Descent Method]
──────────────────────
  Imagine you're on a mountain, it's foggy and you can't see the path
  Goal: Find the valley (minimum loss)
  
  SGD: Look at your feet, take one step down the steepest slope
       Simple but slow, easily stuck on small slopes
  
  Momentum: Slide down with a skateboard
            Has inertia, can roll over small bumps
            Faster speed
  
  Adam: Off-road vehicle with GPS
        Not just looking at feet, but also at historical trajectory
        Intelligently plans the route
  
  Muon: Off-road vehicle with a directional compass
        Not just looking at history, but also 'orthogonalizing' directions
        → No conflicting directions, goes straight and fast


[History of Optimizers]
────────────────

  2012: SGD - Simple, slow
  2015: Adam - Adaptive, fast
  2017: AdamW - Fixed weight decay
  2024: Muon - Orthogonalization, faster

  Trends:
    1. Increasingly complex, but increasingly effective
    2. Adaptive learning rate is the core
    3. Memory and speed tradeoffs


[Diagram: Optimizer Convergence Speed Comparison]
─────────────────────────

  Loss│
      │╲
      │ ╲ AdamW
      │  ╲
      │   ╲  Muon  (faster)
      │    ╲
      │     ╲_____
      │      ╲__  Adam
      │         ╲___
      │             ╲_____  SGD (slowest)
      └──────────────────── step


[Muon's Core: Newton-Schulz Orthogonalization]
─────────────────────────────────

  Why is it effective?
  
  Standard gradient descent:
    W -= lr * grad
    → Gradient direction may be very "flat" (ellipsoid), walks in Z pattern
  
  Muon:
    1. First apply momentum (accumulate velocity)
    2. Newton-Schulz iteration: 'Orthogonalize' the momentum matrix
    3. Update parameters
  
  Effect:
    Matrix singular values all approach 1
    → Each direction progresses at the same speed
    → Won't deviate due to ill-conditioned matrix


[Key Difference Between Adam vs AdamW]
──────────────────────────

  Adam (2015):
    weight_decay is directly added to gradient:
      grad += weight_decay * W
    → Weight decay coupled with learning rate, not ideal
  
  AdamW (2017):
    Weight decay is directly applied to parameters:
      W -= lr * (grad + weight_decay * W)
    → Decoupled, more stable, current mainstream

  Practical effect:
    Adam generalizes poorly on large datasets
    AdamW significantly improves, approaching SGD generalization


[Optimizer Selection Guide]
────────────────

  ┌─────────────────┬──────────────────────┐
  │ Scenario        │ Recommendation       │
  ├─────────────────┼──────────────────────┤
  │ LLM pretraining │ AdamW or Muon+AdamW  │
  │ LLM fine-tuning │ AdamW                │
  │ Computer vision │ SGD or AdamW         │
  │ Reinforcement   │ Adam (fast, allows   │
  │ learning        │ overfitting)         │
  │ Small datasets  │ SGD (good            │
  │                 │ generalization)      │
  │ Quick           │ AdamW (stable)       │
  │ experiments     │                      │
  └─────────────────┴──────────────────────┘


[MiniMind's Optimizer Configuration]
──────────────────────

  Pretraining:
    optimizer = AdamW(lr=1e-4, weight_decay=0.01)
    + warmup + cosine decay
  
  Advanced:
    optimizer = Muon(2D params) + AdamW(others)
    → Significantly saves memory, faster convergence
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] SGD vs Adam
  For simple problems (linear regression), which converges faster: SGD or Adam?
  What about complex problems (deep networks)?

[Exercise 2] Learning Rate Selection
  What is the recommended learning rate for Adam? For SGD? Why is there such a big difference?

[Exercise 3] AdamW Advantage
  What is the core improvement of AdamW over Adam?

[Exercise 4] Muon Applicability
  Is Muon suitable for all parameters? Which parameters cannot use Muon?

[Exercise 5] Mixed Optimizer
  How to allocate parameters when training LLaMA-7B with Muon + AdamW mixed?

[Exercise 6] Memory Comparison
  How much extra memory do AdamW and SGD need per parameter?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
print("  Simple problems (linear regression):")
print("    SGD is faster, converges almost instantly")
print("    Adam's adaptivity is actually a burden (needs warmup)")
print()
print("  Complex problems (deep networks):")
print("    Adam significantly outperforms SGD")
print("    Reasons:")
print("      - Deep network loss landscape is complex")
print("      - Different parameters need different learning rates")
print("      - Adam's adaptivity is just right")
print()
print("  Conclusion:")
print("    There is no 'optimal' optimizer, only 'suitable' ones")
print("    Simple tasks use SGD, complex tasks use Adam")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Adam recommended lr: 1e-3 to 1e-4")
print("  SGD recommended lr: 1e-1 to 1e-2")
print()
print("  100x difference, why?")
print()
print("  SGD:")
print("    Uses gradients directly, gradients can be large")
print("    Needs a larger learning rate to take sufficient steps")
print("    But too large causes oscillation")
print()
print("  Adam:")
print("    Gradients are 'normalized' by second moment")
print("    update = grad / sqrt(var)")
print("    → Effective step size is O(1), independent of gradient magnitude")
print("    → So a very small lr can be used")
print()
print("  Experiment:")
print("    Adam lr=1e-3 ≈ SGD lr=1e-1 step size")
print("    But convergence speed differs by many times")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Core improvement: Decoupled weight decay")
print()
print("  Adam (old):")
print("    grad = original gradient + weight_decay * W")
print("    W = W - lr * grad")
print("    → weight_decay effect depends on lr")
print("    → When lr decreases, weight_decay decreases proportionally, unreasonable")
print()
print("  AdamW (new):")
print("    grad = original gradient  (no weight_decay)")
print("    W = W - lr * (grad + weight_decay * W)")
print("    → weight_decay effect is independent of lr")
print("    → More stable tuning, better generalization")
print()
print("  Experimental evidence:")
print("    Training ResNet:")
print("      Adam:  78% accuracy")
print("      AdamW: 79.5% accuracy")
print("    Difference seems small, but significant on large models")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  Muon suitable for: 2D parameters (matrices)")
print("  Muon not suitable for: 1D parameters (vectors)")
print()
print("  Reason:")
print("    Newton-Schulz orthogonalization acts on matrices")
print("    Vectors have no concept of 'orthogonalization'")
print()
print("  LLM parameter classification:")
print("    2D (use Muon):")
print("      - q_proj.weight: [hidden, hidden]")
print("      - k_proj.weight")
print("      - v_proj.weight")
print("      - o_proj.weight")
print("      - gate_proj, up_proj, down_proj.weight")
print("      - embedding.weight (can also be viewed as 2D)")
print("    1D (use AdamW):")
print("      - layernorm.weight")
print("      - layernorm.bias")
print("      - all biases")
print()
print("  MiniMind configuration example:")
print("    muon_params = [p for p in model.parameters() if p.ndim >= 2]")
print("    adamw_params = [p for p in model.parameters() if p.ndim < 2]")
print("    optimizer = MixedOptimizer(")
print("        Muon(muon_params, lr=0.02),")
print("        AdamW(adamw_params, lr=3e-4),")
print("    )")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Typical allocation (Kimi/Moonlight and other large models):")
print()
print("    Muon:  All 2D matrix parameters (attention + FFN)")
print("    AdamW: All 1D parameters (norm + bias)")
print("           + embedding")
print("           + lm_head")
print()
print("  Reason:")
print("    - Muon works best on 2D matrices")
print("    - Embedding and LM Head are 'lookups', not transformations")
print("    - Don't need orthogonalization, AdamW is actually better")
print()
print("  Learning rate ratio:")
print("    Muon:   lr = 0.02")
print("    AdamW:  lr = 3e-4 (60x smaller)")
print()
print("  Practical effect:")
print("    - Training speed improvement 1.5-2x")
print("    - Memory savings ~50% (Muon doesn't store second moment)")
print("    - Final loss 5-10% lower")

# Exercise 6
print("\n[Exercise 6 Answer]")
print("  SGD:")
print("    Extra memory per parameter: 0 bytes")
print("    (Only uses the gradient itself)")
print()
print("  Adam/AdamW:")
print("    Extra memory per parameter: 8 bytes (FP32 state)")
print("    - First moment m: 4 bytes")
print("    - Second moment v: 4 bytes")
print()
print("  Muon:")
print("    Extra memory per parameter: 4 bytes (FP32 momentum)")
print("    - Only stores first moment")
print("    - Second moment is computed 'online' by NS iteration")
print()
print("  Comparison (1 billion parameter model):")
print("    SGD:    0 GB extra")
print("    Muon:   4 GB extra")
print("    AdamW:  8 GB extra")
print()
print("  → Muon saves 50% optimizer memory compared to AdamW")
print("  → This is why large model training prefers Muon")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)
