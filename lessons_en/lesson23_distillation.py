# ============================================================================
# Lesson 23: Knowledge Distillation
# ============================================================================
"""
Large models are powerful but slow; small models are fast but weak.
Knowledge distillation: Let a small model (student) learn from a large model (teacher).

Topics:
1. Why Knowledge Distillation?
2. Core Idea: Soft Labels vs Hard Labels
3. KL Divergence & Distillation Loss
4. Temperature Parameter
5. Distillation Training Loop

Key Concepts:
- Teacher Model: Large model providing soft labels
- Student Model: Small model learning soft labels
- Soft Labels: Probability distributions from teacher output
- Hard Labels: One-hot encoded ground truth
- Temperature: Controls "softness" of soft labels
- KL Divergence: Measures difference between two distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. Why Knowledge Distillation?
# ============================================================================
"""
Problem:
  - GPT-4 is powerful but expensive to run (hundreds of billions of params)
  - Deploying to mobile/edge requires small models (hundreds of millions)
  - Training small models directly yields poor results

Solution: Knowledge Distillation
  Use the large model (teacher) output as "soft labels" to train the small model (student)

Intuition:
  Traditional training: Student only sees the answer (hard label)
    "This image is a cat" → [0, 0, 1, 0, 0]  (one-hot)

  Knowledge distillation: Student sees teacher's scoring (soft label)
    "This image is 90% cat, 5% dog, 3% tiger..." → [0.05, 0.03, 0.90, 0.01, 0.01]

  Soft labels contain more information! They tell the student "cats and tigers are similar"
  — this is Dark Knowledge.

[Why are soft labels better than hard labels? A concrete example]

  Suppose we're doing image classification with 5 classes: cat, dog, tiger, car, table

  Hard labels (traditional training):
    Correct answer is "cat" -> [0, 0, 1, 0, 0]
    Student only knows: "The answer is cat, everything else is wrong"
    Learned: cat != dog, cat != car (but not which ones are *similar* to cat)

  Soft labels (knowledge distillation, T=2):
    Teacher output -> [0.60, 0.10, 0.25, 0.03, 0.02]
    Student learns:
      - Cat has highest probability (60%) -> answer is cat ✓
      - Tiger is second (25%) -> cat and tiger are similar! (both felines)
      - Dog is third (10%) -> cat and dog are somewhat similar! (both pets)
      - Car and table have very low probability -> cat is nothing like them

  This is "dark knowledge": hard labels only tell you "which is correct",
  soft labels also tell you "what is similar to what" — this similarity info is invaluable!

  Analogy:
    Hard label = exam only tells you "the answer is C"
    Soft label = teacher says "C is most correct, but B has some merit, A is completely wrong"
    -> Soft labels help you understand "why", not just "what"
"""

print("=" * 70)
print("Lesson 23: Knowledge Distillation")
print("=" * 70)

# ============================================================================
# 2. Soft Labels vs Hard Labels
# ============================================================================
"""
Hard Label:
  True class = 2 → [0, 0, 1, 0, 0, 0]
  Only the correct class is 1, rest are 0, no extra information

Soft Label:
  Teacher output → [0.05, 0.03, 0.85, 0.04, 0.02, 0.01]
  Correct class has highest probability, but other classes have non-zero probs
  These non-zero probabilities encode inter-class similarity (dark knowledge)

Temperature effect:
  softmax(z/T) — larger T → softer (more uniform) distribution

  T=1:  [0.05, 0.03, 0.85, 0.04, 0.02, 0.01]  ← normal softmax
  T=2:  [0.12, 0.09, 0.45, 0.14, 0.11, 0.09]  ← softer, dark knowledge more visible
  T=10: [0.16, 0.16, 0.19, 0.17, 0.16, 0.16]  ← very soft, near uniform
"""

print("\n--- Soft Labels vs Hard Labels ---")
logits = torch.tensor([2.0, 1.0, 5.0, 1.5, 0.5, 0.3])

hard_label = F.one_hot(torch.tensor([2]), num_classes=6).float()[0]
print(f"Hard label: {hard_label.tolist()}")

for T in [1, 2, 5, 10]:
    soft_label = F.softmax(logits / T, dim=0)
    print(f"Soft label (T={T:2d}): {[f'{p:.3f}' for p in soft_label.tolist()]}")

# ============================================================================
# 3. KL Divergence & Distillation Loss
# ============================================================================


def distillation_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.5):
    """Knowledge Distillation Loss

    L = α * L_hard + (1-α) * L_soft

    L_hard = CrossEntropy(student_logits, labels)
    L_soft = KL(student_soft || teacher_soft) * T²

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Distillation temperature
        alpha: Weight for hard label loss (0~1)

    Returns:
        loss: Distillation loss
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    soft_loss = (temperature ** 2) * kl_loss

    return soft_loss


def full_distillation_loss(student_logits, teacher_logits, labels,
                           temperature=1.0, alpha=0.5):
    """Full distillation loss (including hard label loss)

    In LM distillation, typically only soft_loss is used (alpha=0),
    since the "hard label" cross-entropy is already covered in pretraining.
    """
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    soft_loss = distillation_loss(student_logits, teacher_logits, temperature)
    return alpha * hard_loss + (1 - alpha) * soft_loss


# Demo distillation loss
print("\n--- Distillation Loss Calculation ---")
torch.manual_seed(42)
B, T, V = 2, 8, 100

student_logits = torch.randn(B, T, V)
teacher_logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))

for T_val in [1, 2, 4, 8]:
    loss = distillation_loss(student_logits, teacher_logits, temperature=T_val)
    print(f"  T={T_val}: soft_loss = {loss.item():.4f}")

# ============================================================================
# 4. Temperature Deep Dive
# ============================================================================
"""
Why multiply by T²?
  softmax(z/T) gradients are T times smaller than softmax(z)
  Multiplying by T² ensures gradient magnitude is temperature-independent
  Math: ∂L/∂z ∝ 1/T, so we need * T² to compensate
"""

print("\n--- Temperature Impact ---")
logits = torch.tensor([3.0, 1.0, 0.5, -1.0, -2.0])

for T in [0.5, 1, 2, 5, 10]:
    probs = F.softmax(logits / T, dim=0)
    entropy = -(probs * probs.log()).sum()
    print(f"  T={T:4.1f}: entropy={entropy.item():.3f}, probs={[f'{p:.3f}' for p in probs.tolist()]}")

print("  Higher entropy → more uniform distribution → more visible dark knowledge")

# ============================================================================
# 5. Distillation Training Loop
# ============================================================================


class TeacherModel(nn.Module):
    """Teacher Model (large)"""

    def __init__(self, vocab_size=100, dim=128, n_layers=4, n_heads=4, max_len=32):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


class StudentModel(nn.Module):
    """Student Model (small)"""

    def __init__(self, vocab_size=100, dim=32, n_layers=2, n_heads=2, max_len=32):
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
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def train_distillation(teacher, student, dataset, epochs=5, lr=3e-4,
                       temperature=2.0, alpha=0.0):
    """Knowledge Distillation Training Loop

    Args:
        teacher: Teacher model (frozen)
        student: Student model (training)
        temperature: Distillation temperature
        alpha: Hard label loss weight (0=pure distillation, 1=pure hard labels)
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    teacher.eval()

    student.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for input_ids, labels in loader:
            with torch.no_grad():
                teacher_logits = teacher(input_ids)

            student_logits = student(input_ids)

            if alpha > 0:
                loss = full_distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=temperature, alpha=alpha
                )
            else:
                loss = distillation_loss(
                    student_logits, teacher_logits, temperature=temperature
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}, Distill Loss: {avg_loss:.4f}")

    return student


class SimpleDistillDataset:
    def __init__(self, vocab_size=100, max_length=32, n_samples=20):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        input_ids = torch.randint(3, self.vocab_size, (self.max_length,))
        labels = input_ids.clone()
        labels[:10] = -100
        return input_ids, labels


# Run distillation training
print("\n--- Knowledge Distillation Training Demo ---")
teacher = TeacherModel(vocab_size=100, dim=128, n_layers=4, n_heads=4, max_len=32)
student = StudentModel(vocab_size=100, dim=32, n_layers=2, n_heads=2, max_len=32)
teacher.requires_grad_(False)

teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e3
student_params = sum(p.numel() for p in student.parameters()) / 1e3
print(f"Teacher params: {teacher_params:.1f}K")
print(f"Student params: {student_params:.1f}K")
print(f"Compression ratio: {teacher_params / student_params:.1f}x")

distill_dataset = SimpleDistillDataset(vocab_size=100, max_length=32, n_samples=20)

print("\n[Distillation Training Start] T=2.0, α=0.0 (pure distillation)")
student = train_distillation(teacher, student, distill_dataset,
                             epochs=5, lr=3e-4, temperature=2.0, alpha=0.0)

# ============================================================================
# 6. Distillation Strategies Comparison
# ============================================================================
print("\n--- Distillation Strategies ---")
print("""
┌──────────────────┬──────────────────────────────────────────┐
│ Strategy         │ Description                              │
├──────────────────┼──────────────────────────────────────────┤
│ Pure distill (α=0) │ Only teacher soft labels              │
│ Pure hard (α=1)  │ Only ground truth labels                 │
│ Mixed (α=0.5)    │ Soft + hard combined, usually best       │
│ Feature distill  │ Align intermediate layer features        │
│ Progressive      │ Gradually decrease temperature           │
└──────────────────┴──────────────────────────────────────────┘
""")

# ============================================================================
# Exercises
# ============================================================================
print("\n" + "=" * 70)
print("Exercises")
print("=" * 70)

# Exercise 1: Manual KL divergence
print("\nExercise 1: Manual KL divergence")
p = torch.tensor([0.8, 0.1, 0.1])
q = torch.tensor([0.4, 0.3, 0.3])
kl = F.kl_div(q.log(), p, reduction='sum')
print(f"Teacher P = {p.tolist()}")
print(f"Student Q = {q.tolist()}")
print(f"KL(P||Q) = {kl.item():.4f}")
print("Note: KL divergence is asymmetric! KL(P||Q) ≠ KL(Q||P)")

# Exercise 2: Temperature effect on soft labels
print("\nExercise 2: Temperature effect")
logits = torch.tensor([5.0, 1.0, 0.0, -1.0])
for T in [1, 2, 5]:
    soft = F.softmax(logits / T, dim=0)
    print(f"  T={T}: {soft.tolist()}")

# Exercise 3: Why T²
print("\nExercise 3: Why multiply by T²?")
print("Answer: softmax(z/T) gradients are T times smaller than softmax(z),")
print("      multiplying by T² keeps gradient magnitude temperature-independent.")

# Exercise 4: Compression ratio
print("\nExercise 4: Model compression ratio")
print(f"Teacher: dim=128, layers=4, heads=4 → {teacher_params:.1f}K params")
print(f"Student: dim=32, layers=2, heads=2 → {student_params:.1f}K params")
print(f"Compression: {teacher_params / student_params:.1f}x")
print("For 10x compression: params ∝ dim² * layers, so halving dim ≈ 4x, halving layers ≈ 2x")

print("\n" + "=" * 70)
print("Lesson 23 Summary:")
print("  1. Distillation: Small model (student) learns from large model (teacher)")
print("  2. Soft labels contain dark knowledge (inter-class similarity)")
print("  3. Temperature T controls soft label 'softness', larger T → more uniform")
print("  4. Distillation loss = KL(teacher || student) * T²")
print("  5. Strategies: pure distillation, mixed, feature-level, progressive")
print("=" * 70)
