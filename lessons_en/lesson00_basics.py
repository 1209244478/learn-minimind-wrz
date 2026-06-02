"""
Lesson 0: Prerequisites — Getting Started
==========================================

Before diving into LLMs, you need three fundamental skills:
  1. Python basics
  2. PyTorch tensor operations
  3. A bit of linear algebra

This lesson covers no model knowledge — just the prerequisites.
If you already know these, feel free to skip.

Run: python lessons_en/lesson00_basics.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Part 1: Python Basics
# ============================================================

print("=" * 60)
print("Part 1: Python Basics")
print("=" * 60)

print("""
Python is a clean, readable programming language. Let's quickly review what you need to know.
""")

# 1. Variables and data types
print("[1.1] Variables and Data Types")
print("-" * 40)

name = "MiniMind"      # string str
age = 1                # integer int
pi = 3.14              # float
is_active = True       # boolean bool
nothing = None         # NoneType

print(f"String:  {name} (type: {type(name).__name__})")
print(f"Integer: {age} (type: {type(age).__name__})")
print(f"Float:   {pi} (type: {type(pi).__name__})")
print(f"Boolean: {is_active} (type: {type(is_active).__name__})")
print(f"None:    {nothing} (type: {type(nothing).__name__})")


# 2. Lists and dictionaries
print("\n[1.2] Lists and Dictionaries")
print("-" * 40)

tokens = [1, 5, 10, 15, 20]
print(f"tokens = {tokens}")
print(f"First element:  tokens[0] = {tokens[0]}")
print(f"Last element:   tokens[-1] = {tokens[-1]}")
print(f"Slice tokens[1:3] = {tokens[1:3]}")
print(f"Length: len(tokens) = {len(tokens)}")

# Append element
tokens.append(25)
print(f"After append: {tokens}")

# Dictionary: key-value pairs
config = {
    "hidden_size": 512,
    "num_layers": 8,
    "vocab_size": 6400,
}
print(f"\nconfig = {config}")
print(f"Hidden size: config['hidden_size'] = {config['hidden_size']}")
print(f"All keys: {list(config.keys())}")


# 3. Loops and conditionals
print("\n[1.3] Loops (for) and Conditionals (if)")
print("-" * 40)

print("for loop over list:")
for token in tokens:
    print(f"  token = {token}")

print("\nfor loop with index:")
for i, token in enumerate(tokens[:3]):
    print(f"  index {i}: {token}")

print("\nif conditional:")
score = 0.95
if score > 0.9:
    print(f"  Score {score} > 0.9, the model is confident!")
elif score > 0.5:
    print(f"  Score {score} > 0.5, the model is somewhat sure")
else:
    print(f"  Score {score} is too low, the model is uncertain")


# 4. Functions
print("\n[1.4] Functions (def)")
print("-" * 40)

def greet(name, language="en"):
    """Greeting function - docstring explains purpose"""
    if language == "en":
        return f"Hello, {name}!"
    else:
        return f"Ni hao, {name}!"

print(greet("Alice"))
print(greet("Alice", language="zh"))


# 5. Classes
print("\n[1.5] Classes - Object-Oriented Programming")
print("-" * 40)

class SimpleTokenizer:
    """A simple tokenizer - demonstrates how to write a class"""

    def __init__(self, vocab):
        """Initialize: build vocabulary"""
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}

    def encode(self, text):
        """Text -> token IDs"""
        return [self.token_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        """Token IDs -> text"""
        return "".join([self.vocab[i] for i in ids])

vocab = ["<pad>", "H", "e", "l", "o", " "]
tokenizer = SimpleTokenizer(vocab)

ids = tokenizer.encode("Hello")
print(f"Encode: Hello -> {ids}")
print(f"Decode: {ids} -> '{tokenizer.decode(ids)}'")


# ============================================================
# Part 2: PyTorch Basics
# ============================================================

print("\n" + "=" * 60)
print("Part 2: PyTorch Basics")
print("=" * 60)

print("""
PyTorch is the most popular deep learning framework. Its core data structure is the "Tensor".
Think of it as a multi-dimensional array, similar to NumPy, but it runs on GPU and supports automatic differentiation.
""")


# 1. Creating tensors
print("[2.1] Creating Tensors")
print("-" * 40)

# From Python list
t1 = torch.tensor([1, 2, 3, 4])
print(f"From list: {t1}, shape={t1.shape}, dtype={t1.dtype}")

# All zeros
t2 = torch.zeros(2, 3)
print(f"Zeros:\n{t2}\nshape={t2.shape}")

# All ones
t3 = torch.ones(2, 3)
print(f"Ones:\n{t3}\nshape={t3.shape}")

# Random
t4 = torch.randn(2, 3)
print(f"Random (normal):\n{t4}\nshape={t4.shape}")

# Special
t5 = torch.arange(0, 6).reshape(2, 3)  # 0,1,2,3,4,5 arranged as 2x3
print(f"arange+reshape:\n{t5}")


# 2. Tensor properties
print("\n[2.2] Important Tensor Properties")
print("-" * 40)

x = torch.randn(2, 3, 4)
print(f"x.shape = {x.shape}    # shape: (2, 3, 4)")
print(f"x.ndim  = {x.ndim}     # number of dims: 3")
print(f"x.dtype = {x.dtype}   # data type: float32")
print(f"x.device = {x.device}  # which device (CPU/GPU)")

# Dimension meanings
print("""
Dimension meanings:
  1D = vector: [1, 2, 3]                   (features of one word)
  2D = matrix: [[1,2], [3,4]]              (multiple words in a sentence)
  3D = tensor: [[[...]]]                   (multiple sentences in a batch)
  4D = [batch, seq, heads, dim]            (multi-head attention)
""")


# 3. Tensor operations
print("[2.3] Tensor Operations")
print("-" * 40)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"a = \n{a}")
print(f"b = \n{b}")
print(f"\nElement-wise add:  a + b = \n{a + b}")
print(f"Element-wise mul:  a * b = \n{a * b}")
print(f"Matrix multiply:   a @ b = \n{a @ b}")
print(f"Scalar add:        a + 10 = \n{a + 10}")


# 4. Indexing and slicing
print("\n[2.4] Indexing and Slicing")
print("-" * 40)

x = torch.arange(12).reshape(3, 4)
print(f"x = \n{x}")
print(f"x[0, :]    = {x[0, :]}    # row 0")
print(f"x[:, 1]    = {x[:, 1]}    # column 1")
print(f"x[1:, 2:]  = \n{x[1:, 2:]}  # bottom-right 2x2")
print(f"x[0, 0]    = {x[0, 0]}    # single element")


# 5. Shape manipulation
print("\n[2.5] Shape Manipulation (reshape, transpose, permute)")
print("-" * 40)

x = torch.arange(24)
print(f"Original: x.shape = {x.shape}")

y = x.reshape(2, 3, 4)
print(f"reshape(2,3,4): y.shape = {y.shape}")

z = y.transpose(0, 1)  # swap dim 0 and dim 1
print(f"transpose(0,1): z.shape = {z.shape}")

print("""
Common operations:
  reshape:    change shape, data unchanged (total elements must match)
  transpose:  swap two dimensions
  permute:    rearrange all dimensions
  squeeze:    remove dimensions of size 1
  unsqueeze:  add a dimension of size 1
""")


# 6. Broadcasting
print("[2.6] Broadcasting")
print("-" * 40)

a = torch.tensor([[1], [2], [3]])  # shape: (3, 1)
b = torch.tensor([10, 20, 30])     # shape: (3,)
c = a + b  # broadcast: b is replicated 3 times to become (3, 3)
print(f"a (3,1) + b (3,) =\n{c}\nshape={c.shape}")

print("""
Broadcasting rules:
  1. Align dimensions from right to left
  2. Dimensions match or one of them is 1 -> can broadcast
  3. Dimensions of size 1 are "replicated" to match
""")


# 7. Automatic differentiation
print("\n[2.7] Automatic Differentiation (Autograd)")
print("-" * 40)

print("""
The core of neural networks is "automatic differentiation":
  1. Create parameters (requires_grad=True)
  2. Forward pass to compute loss
  3. loss.backward() automatically computes gradients
  4. optimizer.step() updates parameters
""")

# Demo
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # y = x^2
y.backward()  # dy/dx = 2x = 4
print(f"x = 2, y = x^2 = {y.item()}")
print(f"dy/dx = 2x = {x.grad.item()}  (theoretical value is 4)")

# Common pattern in neural networks
w = torch.randn(3, 4, requires_grad=True)  # weight
x = torch.randn(2, 3)                       # input
y = x @ w                                   # forward: y = x @ w
loss = y.sum()                              # simple loss
loss.backward()                             # backward
print(f"\nWeight gradient shape: {w.grad.shape}")
print(f"Gradient = the 'correction direction' for weight w")


# 8. Neural network basics
print("\n[2.8] Neural Network Basics (nn.Module)")
print("-" * 40)

print("""
PyTorch uses nn.Module to organize network layers:
  1. Inherit from nn.Module
  2. Define layers in __init__
  3. Define data flow in forward
""")

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)  # input 10-dim, output 20-dim
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)   # input 20-dim, output 5-dim

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = TinyModel()
x = torch.randn(4, 10)  # 4 samples, each 10-dim
y = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# 9. GPU acceleration
print("\n[2.9] GPU Acceleration (optional)")
print("-" * 40)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = x @ y
    print(f"GPU available! Computation done on {device}")
else:
    print("GPU not available, using CPU (fine for learning)")

print("""
Moving data between CPU and GPU:
  .to(device)     - move data to specified device
  .cpu()          - move back to CPU
  .cuda()         - move to GPU (shorthand)
""")


# ============================================================
# Part 3: Linear Algebra Quick Review
# ============================================================

print("\n" + "=" * 60)
print("Part 3: Linear Algebra Quick Review")
print("=" * 60)

print("""
Matrix operations are everywhere in LLMs. Let's quickly review what you need to know.
""")

# 1. Vectors
print("[3.1] Vectors")
print("-" * 40)

v = torch.tensor([1.0, 2.0, 3.0])
print(f"Vector v = {v}")
print(f"Dimension = {v.shape}")
print(f"A word's features are represented as a vector: [0.1, -0.3, 0.5, ...]")


# 2. Matrix multiplication
print("\n[3.2] Matrix Multiplication")
print("-" * 40)

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print(f"A = \n{A}")
print(f"B = \n{B}")
print(f"A @ B = \n{A @ B}")

print("""
Geometric meaning of matrix multiplication:
  - (m, k) @ (k, n) -> (m, n)
  - Inner dimensions must match (k == k)
  - Can be understood as "linear transformation" and "feature combination"

  nn.Linear is essentially matrix multiplication:
    y = x @ W^T + b
  where W is the weight matrix and b is the bias
""")


# 3. Dot product
print("\n[3.3] Dot Product")
print("-" * 40)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
dot = (a * b).sum()
print(f"a = {a}")
print(f"b = {b}")
print(f"Dot product a . b = {dot}")
print(f"  = 1*4 + 2*5 + 3*6 = 4+10+18 = 32")

print("""
Geometric meaning of dot product:
  - Measures "similarity" between two vectors
  - Same direction -> large positive number
  - Opposite direction -> large negative number
  - Perpendicular -> 0

  The core of Attention is the dot product!
  Larger Q . K means Q and K are more related
""")


# 4. Softmax
print("\n[3.4] Softmax: Turning Arbitrary Numbers into Probabilities")
print("-" * 40)

x = torch.tensor([1.0, 2.0, 3.0])
probs = torch.softmax(x, dim=-1)
print(f"Input:    {x}")
print(f"Softmax: {probs}")
print(f"  All values are positive")
print(f"  All values sum to {probs.sum():.4f} = 1.0")
print(f"  Largest input -> largest probability")

print("""
Softmax formula:
  softmax(x_i) = exp(x_i) / sum(exp(x_j))

Why do we need Softmax?
  - Model outputs are arbitrary real numbers (logits)
  - We need to convert them to probabilities (sum to 1)
  - It also "amplifies" the largest value, making the choice clearer
""")


# ============================================================
# Setup Check
# ============================================================

print("\n" + "=" * 60)
print("Setup Check")
print("=" * 60)

print(f"""
  Python version:   {torch.__version__[:6]} (PyTorch)
  PyTorch version:  {torch.__version__}
  CUDA available:   {torch.cuda.is_available()}
""")

print("""
Required tools:
  - Python 3.8+
  - PyTorch 2.0+
  - A text editor (VSCode / PyCharm / Trae)
  - A little patience :)

Optional:
  - GPU (faster training, but CPU works for learning)
  - Jupyter Notebook (interactive execution)

Next up: Lesson 1 - Tokenizer!
""")
