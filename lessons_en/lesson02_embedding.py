"""
Lesson 2: Embedding — How Do Numbers Become Vectors?
=====================================================

Why do we need Embedding?
  Previous lesson: Text -> Numbers (Token IDs)
  This lesson:    Numbers -> Vectors (Embedding)
  Next lesson:    Vectors -> Understanding semantics (Attention)

Why are numbers not enough?
  "King"   = 1234
  "Queen"  = 5678
  "Apple"  = 9012

  Problem: 1234 - 5678 = ? Meaningless
           1234 + 9012 = ? Also meaningless

  Models need to do math, so numbers need "meaning"

Core idea of Embedding:
  Map each token ID to a d-dimensional vector
  Each dimension in the vector represents some "semantic feature"

  Example d=4:
    King   -> [0.8, 0.9, 0.2, 0.1]  (maleness=high, power=high)
    Queen  -> [0.7, 0.2, 0.2, 0.1]  (maleness=low, power=high)
    Princess -> [0.6, 0.3, 0.1, 0.1]  (maleness=low, power=medium)

Analogy:
  Dictionary lookup: given a word, return its definition
  Embedding: given an ID, return a vector
  - Essentially a V x d lookup table
  - V = vocabulary size (vocab_size)
  - d = embedding dimension (hidden_size)

MiniMind uses:
  - vocab_size = 6400
  - hidden_size = 768

Run: python lessons_en/lesson02_embedding.py
"""

import torch
import torch.nn as nn
import math


# ============================================================
# Part 1: Manual Embedding Implementation
# ============================================================
print("=" * 60)
print("Part 1: Manual Embedding (Lookup Table)")
print("=" * 60)


class SimpleEmbedding:
    """Manual Embedding: essentially a lookup table"""

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Random initialization: V x d matrix
        self.weight = torch.randn(vocab_size, embed_dim) * 0.1

    def forward(self, token_ids):
        """
        token_ids: [batch, seq_len] integer IDs
        Returns: [batch, seq_len, embed_dim] vectors
        """
        return self.weight[token_ids]


# Demo
vocab_size = 100
embed_dim = 8
embedding = SimpleEmbedding(vocab_size, embed_dim)

# Assume a batch: 2 sentences, 5 tokens each
token_ids = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

vectors = embedding.forward(token_ids)
print(f"\nInput token_ids shape: {token_ids.shape}")
print(f"Input: \n{token_ids}")
print(f"\nOutput vectors shape: {vectors.shape}")
print(f"Each token becomes a {embed_dim}-dim vector")
print(f"\nFirst sentence's embedding:")
print(vectors[0])


# ============================================================
# Part 2: PyTorch's nn.Embedding
# ============================================================
print("\n" + "=" * 60)
print("Part 2: PyTorch's nn.Embedding")
print("=" * 60)

# Actual usage
embedding = nn.Embedding(
    num_embeddings=100,    # vocabulary size
    embedding_dim=8,       # embedding dimension
    padding_idx=0          # padding index (always zero vector)
)

vectors = embedding(token_ids)
print(f"\nnn.Embedding output shape: {vectors.shape}")
print(f"Parameters: {sum(p.numel() for p in embedding.parameters())}")

# Important: padding_idx is always a zero vector
print(f"\nVector at padding_idx=0 (always zero):")
print(embedding.weight[0])
print(f"All zeros: {torch.all(embedding.weight[0] == 0).item()}")


# ============================================================
# Part 3: Semantics of Embedding
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Semantics of Embedding")
print("=" * 60)

print("""
After training, Embedding vectors "learn" semantics:
  Semantically similar words have similar vectors
  Semantically opposite words have opposite vectors
  Related words cluster in vector space

Classic example (discovered by Word2Vec):
  vec(King) - vec(Man) + vec(Woman) ~ vec(Queen)

Visualization (2D):
  +------------------+
  |  King *          |
  |  Man o           |
  |         Queen <> |
  |                  |
  | Apple [] Orange ^|
  +------------------+

  *o<> in the same region (all "people")
  []^ in the same region (all "fruits")
  King - Queen = Man - Woman (gender difference)

How to measure similarity between two vectors?
  - Cosine similarity: cos_sim(A, B) = (A.B) / (|A|*|B|)
  - Range [-1, 1], 1=identical, 0=unrelated, -1=opposite
""")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)


# Demo cosine similarity
torch.manual_seed(42)
king = torch.tensor([0.8, 0.9, 0.2])
queen = torch.tensor([0.7, 0.2, 0.2])
man = torch.tensor([0.85, 0.95, 0.1])
apple = torch.tensor([0.1, 0.2, 0.9])

print(f"\nCosine similarity demo:")
print(f"  sim(king, queen) = {cosine_similarity(king, queen):.4f}")
print(f"  sim(king, man)   = {cosine_similarity(king, man):.4f}")
print(f"  sim(king, apple) = {cosine_similarity(king, apple):.4f}")
print(f"\nObservation: King is more similar to Queen than to Apple")


# ============================================================
# Part 4: Impact of Embedding Dimension
# ============================================================
print("\n" + "=" * 60)
print("Part 4: Impact of Embedding Dimension")
print("=" * 60)

print("""
Embedding dimension (d) is a key hyperparameter:

d too small (e.g., 64):
  + Fast training, less memory
  - Weak expressiveness, can't separate complex semantics
  - Underfitting with limited training data

d too large (e.g., 4096):
  + Strong expressiveness
  - Slow training, high memory
  - Needs more training data

Empirical formula (for reference):
  d ~ (V ** 0.25) * 4   (V=vocab_size)

MiniMind actual config:
  - MiniMind-Small:  d=512
  - MiniMind:        d=768
  - Large model (GPT-3):  d=12288 (175B)
""")

# Demo
print("\n[Demo] Recommended dimensions for different vocab sizes")
print("-" * 60)
print(f"{'Vocab Size':<12}{'Recommended d':<15}{'Total Params':<15}")
print("-" * 60)
for V in [1000, 10000, 50000, 100000]:
    d = int((V ** 0.25) * 4)
    total = V * d
    print(f"{V:<12}{d:<15}{total:,}")


# ============================================================
# Part 5: Before vs After Training
# ============================================================
print("\n" + "=" * 60)
print("Part 5: Before vs After Training")
print("=" * 60)

print("""
Before training:
  - Random initialization (N(0, 0.02) or similar)
  - No semantics at all
  - Similarity between any two words is close to 0

After training:
  - Learned through backpropagation
  - Semantically similar words cluster
  - "Directional" semantics emerge

Analogy:
  Before training = Dictionary with only words, no definitions
  After training = Dictionary with definitions and word relationships

Key insight:
  Embedding is the starting point for the model to "understand" language
  Much of the model's "knowledge" is stored in Embedding
  - 7B model, 1/3 of parameters in Embedding
  - Vocab 50K x 4096 = 200M parameters
""")


# ============================================================
# Part 6: Weight Tying
# ============================================================
print("\n" + "=" * 60)
print("Part 6: Weight Tying")
print("=" * 60)

print("""
Interesting observation:
  Embedding's function: token ID -> semantic vector
  LM Head's function:  semantic vector -> token probability

  These two are almost "inverse" operations!
  -> Many models (including MiniMind) share weights between them

Benefits:
  + Halves the parameters (Embedding reduced by half)
  + More stable training (symmetric gradients)
  + Often better performance

Code:
```python
self.embed = nn.Linear(vocab_size, d, bias=False)  # Embedding
self.lm_head = nn.Linear(d, vocab_size, bias=False) # LM Head
self.lm_head.weight = self.embed.weight.T           # Share weights
```

Or more commonly:
```python
self.embed = nn.Embedding(vocab_size, d)
self.lm_head = nn.Linear(d, vocab_size, bias=False)
self.lm_head.weight = self.embed.weight             # Direct sharing
```
""")


# ============================================================
# Part 7: Embedding in MiniMind
# ============================================================
print("\n" + "=" * 60)
print("Part 7: Embedding in MiniMind")
print("=" * 60)

print("""
MiniMind's Embedding implementation (from model_minimind.py):

```python
class MiniMindEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)  # [batch, seq_len, hidden]
        return self.dropout(x)
```

Config (from MiniMindConfig):
  vocab_size: 6400
  hidden_size: 768

Full pipeline:
  input_ids: [batch, seq_len] integers
      | Embedding
  x:        [batch, seq_len, 768] float vectors
      | Transformer Blocks x N
  hidden_states: [batch, seq_len, 768]
      | LM Head
  logits: [batch, seq_len, 6400] probability distribution
""")

# [NEW] Demo: Training an Embedding to observe semantic changes
print("\n" + "-" * 50)
print("[Demo: How Embeddings Learn Semantic Meaning]")
print("-" * 50)
print("""
We'll train a simple Embedding on word pairs and watch how
similar words cluster together through training.

Before training: all vectors are random, "cat" and "dog" are not similar
After training: "cat" and "dog" become similar (both are animals)
""")

torch.manual_seed(42)
vocab = ["cat", "dog", "apple", "banana", "eat", "like", "is", "animal", "fruit"]
vocab_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim = 8

# Training data: (word1, word2, label) — 1=similar, 0=dissimilar
training_data = [
    ("cat", "is", "animal"), ("dog", "is", "animal"),
    ("apple", "is", "fruit"), ("banana", "is", "fruit"),
    ("cat", "like", "dog"), ("apple", "like", "banana"),
]

embedding = nn.Embedding(vocab_size, embed_dim)

# Similarity before training
print("Similarity matrix BEFORE training (cosine similarity):")
with torch.no_grad():
    all_ids = torch.arange(vocab_size)
    emb = embedding(all_ids)
    sim = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    for i, w in enumerate(vocab):
        row = "  ".join(f"{sim[i,j]:+.2f}" for j in range(min(5, len(vocab))))
        print(f"  {w:8s}: {row}  ...")

# Simple training loop
optimizer = torch.optim.Adam(embedding.parameters(), lr=0.01)
for epoch in range(100):
    total_loss = 0
    for w1, rel, w2 in training_data:
        v1 = embedding(torch.tensor([vocab_to_idx[w1]]))
        v2 = embedding(torch.tensor([vocab_to_idx[w2]]))
        # Similar words should have high dot product
        loss = -torch.sum(v1 * v2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: loss = {total_loss:.4f}")

# Similarity after training
print("\nSimilarity matrix AFTER training (cosine similarity):")
with torch.no_grad():
    emb = embedding(all_ids)
    sim = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    for i, w in enumerate(vocab):
        row = "  ".join(f"{sim[i,j]:+.2f}" for j in range(min(5, len(vocab))))
        print(f"  {w:8s}: {row}  ...")

print("""
Key observations:
  - "cat" and "dog" now have higher similarity (both are animals)
  - "apple" and "banana" now have higher similarity (both are fruits)
  - "cat" and "apple" remain dissimilar (different categories)
  → Embeddings learn semantic meaning through training!
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Embedding Shape Calculation
  Input: [2, 10] token IDs, vocab_size=1000, embed_dim=64
  What is the output shape? How many parameters?

[Exercise 2] padding_idx
  nn.Embedding(vocab_size=100, padding_idx=0)
  - Its weight[0] is always 0, why is this useful?
  - Will the gradient at this position be updated during training?

[Exercise 3] Cosine Similarity
  vec(cat) = [0.5, 0.3, 0.8]
  vec(dog) = [0.6, 0.2, 0.7]
  vec(car) = [-0.5, 0.9, 0.1]
  Compute similarity for cat-dog, dog-car, cat-car. Which pair is most related?

[Exercise 4] Vocabulary Size Selection
  Common Chinese characters ~3500, plus English + digits + punctuation, roughly how many?
  Why does MiniMind use 6400 instead of 3500?

[Exercise 5] Weight Tying
  Embedding matrix is V x d, LM Head is d x V
  If weights are shared, how many parameters are saved?
  Why is strict sharing often not used in practice?
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
V, d = 1000, 64
input_shape = [2, 10]
output_shape = input_shape + [d]
params = V * d
print(f"  Input shape: {input_shape}")
print(f"  Output shape: {output_shape}")
print(f"  Parameters: V x d = {V} x {d} = {params:,}")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Purpose: padding positions should not affect other tokens")
print("           Making padding embedding 0 represents 'empty'")
print("  Gradient: gradient at padding_idx is forced to 0, never updated")
print("            Prevents padding token from learning any 'meaningful' representation")

# Exercise 3
print("\n[Exercise 3 Answer]")
cat = torch.tensor([0.5, 0.3, 0.8])
dog = torch.tensor([0.6, 0.2, 0.7])
car = torch.tensor([-0.5, 0.9, 0.1])

print(f"  sim(cat, dog) = {cosine_similarity(cat, dog):.4f}")
print(f"  sim(dog, car) = {cosine_similarity(dog, car):.4f}")
print(f"  sim(cat, car) = {cosine_similarity(cat, car):.4f}")
print(f"  Conclusion: cat-dog most related (animals), cat-car least related")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  Common Chinese characters: ~3500")
print("  English: 26 letters x 2 cases = 52")
print("  Digits: 10")
print("  Punctuation: ~30")
print("  BPE subwords: ~3000 (frequent phrases)")
print("  Special tokens: 100 (BOS, EOS, PAD, ...)")
print("  Total: ~6400-7000")
print("  Using 6400 supports BPE subwords, larger than pure character vocabulary")

# Exercise 5
print("\n[Exercise 5 Answer]")
print("  Embedding parameters: V x d")
print("  LM Head parameters: d x V = V x d (same)")
print("  Sharing saves: 50% parameters (V x d)")
print("  LLaMA-7B example: saves ~50M parameters (2.5% of total)")
print("  MiniMind (~26M): saves ~3M parameters (~12% of total)")
print("  Practice: strict sharing puts input/output in same space, slightly worse")
print("            Approximate or partial sharing is more common in practice")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Embedding Essence
   - V x d lookup table
   - token ID -> d-dimensional vector
   - Learns semantics during training

2. Key Concepts
   - vocab_size (V): vocabulary size
   - hidden_size (d): embedding dimension
   - padding_idx: padding position

3. Semantics After Training
   - Semantically similar words have similar vectors
   - King - Queen = Man - Woman
   - Cosine similarity measures relatedness

4. Dimension Selection
   - Too small: insufficient expressiveness
   - Too large: slow training, wasteful
   - Empirical: d ~ (V**0.25) * 4

5. Weight Tying
   - Embedding and LM Head share weights
   - Saves 50% parameters
   - More stable training

Next: Lesson 3 - RMSNorm (Why do we need normalization?)
""")
