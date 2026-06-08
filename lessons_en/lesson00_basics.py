"""
Lesson 0: Prerequisites — Getting Started
==========================================

Before diving into LLMs, you need three fundamental skills:
  1. Python basics
  2. PyTorch tensor operations
  3. A bit of linear algebra

This lesson covers no model knowledge — just the prerequisites.
If you already know these, feel free to skip.

[Environment Setup]
  1. Install Python 3.8+: https://www.python.org/downloads/
     Check "Add Python to PATH" during installation (important!)
  2. Open a terminal (Windows: Win+R → type cmd; Mac: open Terminal app)
  3. Install PyTorch: type pip install torch in the terminal
  4. Install other dependencies: pip install numpy
  5. Verify installation: python -c "import torch; print(torch.__version__)"
  6. Run this lesson: python lessons_en/lesson00_basics.py

  If you have a GPU, install the GPU version of PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
  (cu118 corresponds to CUDA 11.8, choose based on your CUDA version)

  Common issues:
    Q: "'pip' is not recognized" → Python not in PATH, reinstall with PATH option
    Q: "No module named torch" → PyTorch not installed, run pip install torch
    Q: Download too slow → Use a mirror: pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

Run: python lessons_en/lesson00_basics.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Part 0: Deep Learning in 5 Minutes
# ============================================================

print("=" * 60)
print("Part 0: Deep Learning in 5 Minutes")
print("=" * 60)

print("""
[What is Deep Learning? In one sentence]

  Deep Learning = Teaching computers to learn skills automatically by "seeing lots of data"

  Analogy: Teaching a child to recognize cats
    Traditional programming: You write rules "if pointy ears + whiskers + tail → it's a cat"
             Problem: You can't write enough rules, there are always exceptions
    Deep learning: Show the child 10,000 photos of cats, they learn to recognize cats on their own
             Advantage: No rules needed, the model discovers patterns itself

[What is a Large Language Model (LLM)?]

  LLM = A deep learning model that learned to "speak" after reading hundreds of billions of words

  Training process:
    1. Show the model lots of text: "The weather today is very ___"
    2. Model guesses the next word: "good" (rewarded if correct, adjusted if wrong)
    3. Repeat trillions of times → Model learns language patterns
    4. You ask it a question, it can answer like a human

[What do you need to learn?]

  To understand LLMs, you need to know:
    1. Python — The language to communicate with computers (Part 1 of this lesson)
    2. Tensors — How computers store data (Part 2 of this lesson)
    3. Matrix operations — The core of LLM computation (Part 3 of this lesson)
    4. Autograd — The magic that makes models "learn automatically" (Part 2 of this lesson)

  Once you learn these 4 things, you can understand all the following lessons!
""")

# ============================================================
# Part 0.5: What are Terminal and pip?
# ============================================================

print("=" * 60)
print("Part 0.5: What are Terminal and pip?")
print("=" * 60)

print("""
If you've never used a terminal, here's a quick explanation:

[Terminal (Terminal / CMD / PowerShell)]
  It's a text window where you type commands and the computer executes them.
  How to open:
    Windows: Press Win+R, type cmd, press Enter
    Mac: Open the "Terminal" app
    VSCode/Trae: Menu → Terminal → New Terminal

[What is pip?]
  pip = Python's "App Store"
  Just like you use the App Store to install apps on your phone,
  Python uses pip to install "packages" (code libraries written by others).

  Common commands:
    pip install xxx    → Install package xxx
    pip uninstall xxx  → Uninstall package xxx
    pip list           → See all installed packages

[What is import?]
  import = "Bring in" a package written by others into your code

  import torch       → Import PyTorch (deep learning framework)
  import numpy as np → Import NumPy (math library), abbreviated as np

  Analogy:
    pip install = Go to a bookstore and buy a book
    import      = Take the book off the shelf and open it
    Use its functions = Follow the methods in the book
""")


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
""")


# ============================================================
# Part 4: Putting It All Together — Train Your First Model
# ============================================================

print("=" * 60)
print("Part 4: Putting It All Together — Train Your First Model")
print("=" * 60)

print("""
We've learned Python, tensors, matrix operations, autograd, nn.Module...
How do these all fit together? Answer: Train a model!

Below we'll train a "number prediction" model in ~20 lines of code.
This is a miniature version of LLM training — same principles, just 10,000x smaller.
""")

# --- Step 1: Prepare Data ---
print("[4.1] Step 1: Prepare Data")
print("-" * 40)

torch.manual_seed(42)
x_train = torch.randn(100, 1)
y_train = x_train * 2 + 1 + torch.randn(100, 1) * 0.1  # y = 2x + 1 + noise

print(f"Training data: {len(x_train)} samples")
print(f"First 3: x={x_train[:3].squeeze().tolist()}")
print(f"         y={y_train[:3].squeeze().tolist()}")
print(f"Goal: Let the model learn y = 2x + 1")


# --- Step 2: Define Model ---
print("\n[4.2] Step 2: Define Model")
print("-" * 40)

class SimpleModel(nn.Module):
    """Simplest model: 1 input -> 1 output (y = w*x + b)"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
print(f"Model: y = w*x + b")
print(f"Initial params: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")
print(f"(Random values — hasn't learned yet)")


# --- Step 3: Define Loss and Optimizer ---
print("\n[4.3] Step 3: Define Loss and Optimizer")
print("-" * 40)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"Loss function: MSELoss (mean squared error between prediction and truth)")
print(f"Optimizer: SGD (learning rate=0.01)")
print(f"Analogy:")
print(f"  Loss function = exam score (lower is better)")
print(f"  Optimizer = study method (adjust a little each time)")


# --- Step 4: Training Loop ---
print("\n[4.4] Step 4: Training Loop (The Core!)")
print("-" * 40)

print("""
The training loop is 4 steps, repeated over and over:
  1. Forward pass: compute predictions with current parameters
  2. Compute loss: how far are predictions from truth?
  3. Backward pass: compute gradients (which direction to adjust)
  4. Update parameters: adjust in the gradient direction
""")

for epoch in range(200):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, w={w:.4f}, b={b:.4f}")


# --- Step 5: Results ---
print("\n[4.5] Step 5: Results")
print("-" * 40)

w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"Before training: w=random, b=random")
print(f"After training:  w={w:.4f}, b={b:.4f}")
print(f"True values:     w=2.0000, b=1.0000")
print(f"Error:           w off by {abs(w-2):.4f}, b off by {abs(b-1):.4f}")
print(f"-> Model learned y = {w:.2f}x + {b:.2f}, very close to y = 2x + 1!")


# --- Test ---
print("\n[4.6] Test: Give the model a new input")
print("-" * 40)

test_x = torch.tensor([[3.0]])
test_y = model(test_x).item()
true_y = 3.0 * 2 + 1
print(f"Input x = 3.0")
print(f"Model predicts: y = {test_y:.4f}")
print(f"True value:     y = {true_y:.4f}")
print(f"Error: {abs(test_y - true_y):.4f}")


print("""
=============================================================
[Summary: How This Demo Relates to LLMs]
=============================================================

  This tiny model              Large Model (GPT)
  ───────────────              ────────────────
  Input: 1 number              Input: text (hundreds of tokens)
  Output: 1 number             Output: probability of next token
  Parameters: 2 (w, b)         Parameters: billions
  Training data: 100 pairs     Training data: hundreds of billions of words
  Training loop: 200 steps     Training loop: millions of steps
  Loss function: MSELoss       Loss function: CrossEntropyLoss

  But the core process is exactly the same!
    1. Prepare data
    2. Define model
    3. Define loss function and optimizer
    4. Loop: forward -> compute loss -> backward -> update
    5. Test the results

  All subsequent lessons just "add things" to this framework:
    - More complex model architectures (Attention, Transformer, ...)
    - Larger data (text corpora)
    - More training tricks (learning rate scheduling, gradient clipping, ...)

  But the core is always these 5 steps!

Next up: Lesson 1 - Tokenizer (turning text into numbers)!
""")
