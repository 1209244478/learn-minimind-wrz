# ============================================================================
# Lesson 21: Supervised Fine-Tuning (SFT)
# ============================================================================
"""
In the previous lesson, we built a mini LLM from scratch. But a pretrained
model only knows how to "continue text" — it can't hold a conversation.
SFT teaches the model to follow instructions and respond as an assistant.

Topics:
1. Why SFT? Pretrain vs Fine-Tuning
2. SFT Data Format: Chat Templates
3. SFT Label Construction: Loss Masking
4. SFT Training Loop
5. Learning Rate & Training Strategy

Key Concepts:
- Pretrain: Learn statistical patterns of language (continuation ability)
- SFT: Learn to follow instructions (conversation ability)
- Chat Template: Format multi-turn conversations into model-readable text
- Loss Mask: Only compute loss on assistant responses, ignore prompts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ============================================================================
# 1. Why SFT?
# ============================================================================
"""
Pretrained model = Text Continuator:
  Input: "The capital of China is"
  Output: "Beijing, located on the North China Plain..."  ← continuation

SFT model = Conversational Assistant:
  Input: "<|im_start|>user\nWhat is the capital of China?<|im_end|>\n<|im_start|>assistant\n"
  Output: "The capital of China is Beijing."  ← follows instruction

Key Differences:
┌──────────┬──────────────────┬──────────────────┐
│          │ Pretrain         │ SFT              │
├──────────┼──────────────────┼──────────────────┤
│ Goal     │ Learn language   │ Follow instruct.  │
│ Data     │ Plain text       │ Instruction pairs │
│ Loss     │ All tokens       │ Assistant only    │
│ LR       │ Larger (1e-4)    │ Smaller (1e-5)    │
│ Epochs   │ More (3-10)      │ Fewer (1-3)       │
└──────────┴──────────────────┴──────────────────┘
"""

print("=" * 70)
print("Lesson 21: Supervised Fine-Tuning (SFT)")
print("=" * 70)

# ============================================================================
# 2. Chat Template
# ============================================================================
"""
Different models use different chat templates to format conversations.

ChatML format (used by Qwen/MiniMind):
<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
Hello, please introduce yourself.<|im_end|>
<|im_start|>assistant
Hello! I am MiniMind...<|im_end|>

Llama format:
[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>
Hello, please introduce yourself. [/INST] Hello! I am Llama...

Key points:
- Special tokens (<|im_start|>, <|im_end|>) mark role boundaries
- The model only generates content in the assistant section
- System prompts define model behavior
"""


def apply_chat_template(messages, tokenizer_map=None):
    """Simple ChatML template implementation

    Args:
        messages: Conversation list, e.g. [{"role": "user", "content": "Hello"}, ...]
        tokenizer_map: Optional mapping for special tokens

    Returns:
        Formatted conversation text
    """
    BOS = "<|im_start|>"
    EOS = "<|im_end|>\n"

    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{BOS}{role}\n{content}{EOS}")

    return "".join(parts)


# Demo Chat Template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of China?"},
    {"role": "assistant", "content": "The capital of China is Beijing."},
]

formatted = apply_chat_template(messages)
print("\n--- Chat Template Example ---")
print(formatted)

# ============================================================================
# 3. SFT Label Construction — Loss Mask
# ============================================================================
"""
SFT Core: Only compute loss on assistant responses!

Why?
- Prompt parts (system + user) are inputs — the model doesn't need to learn them
- Only assistant responses are what the model should learn to generate

Implementation:
- Tokenize the entire conversation into input_ids
- In labels, set non-assistant parts to -100 (ignore_index)
- Assistant response parts: labels = input_ids (normal loss)

Example:
  input_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  labels:    [-100,-100,-100,-100, 5, 6, 7, 8, 9, 10]
              ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
              system+user (ignore)   assistant (learn)
"""


def create_sft_labels(input_ids, bos_assistant_ids, eos_ids, pad_token_id=0):
    """Create labels for SFT data — only compute loss on assistant parts

    Args:
        input_ids: Token ID list for the full conversation
        bos_assistant_ids: Token IDs for "<|im_start|>assistant\n"
        eos_ids: Token IDs for "<|im_end|>"
        pad_token_id: Padding token ID

    Returns:
        labels: Same length as input_ids, non-assistant parts set to -100
    """
    labels = [-100] * len(input_ids)

    i = 0
    while i < len(input_ids):
        # Find assistant response start position
        if input_ids[i:i + len(bos_assistant_ids)] == bos_assistant_ids:
            start = i + len(bos_assistant_ids)
            end = start
            # Find response end position
            while end < len(input_ids):
                if input_ids[end:end + len(eos_ids)] == eos_ids:
                    break
                end += 1
            # Mark assistant part labels
            for j in range(start, min(end + len(eos_ids), len(input_ids))):
                labels[j] = input_ids[j]
            i = end + len(eos_ids) if end < len(input_ids) else len(input_ids)
        else:
            i += 1

    # Padding parts also don't compute loss
    for i in range(len(input_ids)):
        if input_ids[i] == pad_token_id:
            labels[i] = -100

    return labels


# Demo label construction
print("\n--- SFT Label Construction Example ---")
simple_input = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 0, 0]
labels = create_sft_labels(simple_input, bos_assistant_ids=[30], eos_ids=[34], pad_token_id=0)

print(f"input_ids: {simple_input}")
print(f"labels:    {labels}")
print(f"Positions with loss: {[i for i, l in enumerate(labels) if l != -100]}")

# ============================================================================
# 4. SFT Dataset
# ============================================================================


class SimpleSFTDataset:
    """Simplified SFT Dataset

    In practice, data is loaded from JSONL files with format:
    {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """

    def __init__(self, data, vocab_size=100, max_length=64):
        self.data = data
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_assistant_ids = [50, 51]
        self.eos_ids = [2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        torch.manual_seed(idx)
        length = min(len(sample["content"]) + 10, self.max_length)
        input_ids = torch.randint(3, self.vocab_size, (length,)).tolist()
        input_ids = input_ids[:length - 6] + self.bos_assistant_ids + [60, 61, 62] + self.eos_ids
        input_ids = input_ids + [self.pad_token_id] * (self.max_length - len(input_ids))
        input_ids = input_ids[:self.max_length]
        labels = create_sft_labels(input_ids, self.bos_assistant_ids, self.eos_ids, self.pad_token_id)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


sft_data = [
    {"role": "user", "content": "What is machine learning?", "answer": "ML is a branch of AI..."},
    {"role": "user", "content": "What is Python?", "answer": "Python is a programming language..."},
    {"role": "user", "content": "Deep learning vs machine learning?", "answer": "DL is a subset of ML..."},
]

sft_dataset = SimpleSFTDataset(sft_data)
input_ids, labels = sft_dataset[0]
print(f"\n--- SFT Dataset Example ---")
print(f"input_ids shape: {input_ids.shape}")
print(f"labels shape: {labels.shape}")
print(f"Tokens with loss: {(labels != -100).sum().item()}")
print(f"Total tokens: {labels.shape[0]}")

# ============================================================================
# 5. SFT Training Loop
# ============================================================================


class SimpleGPT(nn.Module):
    """Simplified GPT model (for SFT training demo)"""

    def __init__(self, vocab_size=100, dim=64, n_layers=2, n_heads=4, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
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
        self.tok_emb.weight = self.lm_head.weight  # Weight sharing

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Key: ignore_index=-100 makes non-assistant parts skip loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100  # SFT core!
            )

        return logits, loss


def train_sft(model, dataset, epochs=3, lr=1e-5, batch_size=4):
    """SFT Training Loop

    Key differences from pretraining:
    1. Smaller learning rate (1e-5 vs 1e-4)
    2. Fewer epochs (1-3 vs 3-10)
    3. Loss only on assistant parts
    4. Start from pretrained weights (not random init)
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for input_ids, labels in loader:
            logits, loss = model(input_ids, targets=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


# Run SFT training
print("\n--- SFT Training Demo ---")
model = SimpleGPT(vocab_size=100, dim=64, n_layers=2, n_heads=4, max_len=64)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")

print("\n[SFT Training Start]")
model = train_sft(model, sft_dataset, epochs=5, lr=1e-5, batch_size=2)

# ============================================================================
# 6. SFT vs Pretrain Loss Comparison
# ============================================================================
print("\n--- SFT vs Pretrain Loss Comparison ---")

torch.manual_seed(42)
B, T, V = 2, 16, 100
logits = torch.randn(B, T, V)

# Pretrain: all tokens compute loss
pretrain_labels = torch.randint(0, V, (B, T))
pretrain_loss = F.cross_entropy(logits.view(-1, V), pretrain_labels.view(-1))
print(f"Pretrain loss (all tokens): {pretrain_loss.item():.4f}")

# SFT: only assistant tokens compute loss
sft_labels = pretrain_labels.clone()
sft_labels[:, :10] = -100
sft_loss = F.cross_entropy(logits.view(-1, V), sft_labels.view(-1), ignore_index=-100)
n_valid = (sft_labels != -100).sum().item()
n_total = sft_labels.numel()
print(f"SFT loss (assistant only): {sft_loss.item():.4f}")
print(f"Tokens with loss: {n_valid}/{n_total} ({n_valid / n_total * 100:.0f}%)")

# ============================================================================
# 7. Key SFT Training Tips
# ============================================================================
"""
1. Small learning rate (1e-5 ~ 5e-5)
   - Pretrained weights are precious, large LR causes "catastrophic forgetting"

2. Few epochs (1-3)
   - SFT data is much smaller than pretrain data
   - Too many epochs causes overfitting

3. Cosine annealing LR schedule
   - Warm up to peak, then slowly decay

4. Gradient clipping
   - Prevent gradient explosion, typically clip=1.0

5. Weight decay
   - AdamW weight_decay=0.01, prevents overfitting

6. Start from pretrained weights
   - NEVER start SFT from random initialization!
   - Load pretrained weights, then fine-tune with small LR
"""

# ============================================================================
# Exercises
# ============================================================================
print("\n" + "=" * 70)
print("Exercises")
print("=" * 70)

# Exercise 1: Construct SFT labels
print("\nExercise 1: Construct SFT labels")
test_input = [1, 5, 6, 7, 50, 51, 30, 31, 32, 2, 0, 0]
print(f"input_ids: {test_input}")
print(f"BOS_AST = [50, 51], EOS = [2], PAD = 0")
test_labels = create_sft_labels(test_input, [50, 51], [2], 0)
print(f"Answer labels: {test_labels}")
print(f"Verify: loss positions = {[i for i, l in enumerate(test_labels) if l != -100]}")

# Exercise 2: Compute SFT loss
print("\nExercise 2: Compute SFT loss")
torch.manual_seed(0)
logits = torch.randn(1, 8, 50)
labels = torch.tensor([[-100, -100, -100, 10, 20, 30, 2, -100]])
loss = F.cross_entropy(logits.view(-1, 50), labels.view(-1), ignore_index=-100)
print(f"Given logits shape = {logits.shape}, labels = {labels.tolist()}")
print(f"SFT loss = {loss.item():.4f}")
print(f"Tokens with loss = {(labels != -100).sum().item()}")

# Exercise 3: Learning rate impact
print("\nExercise 3: Impact of learning rate on SFT")
print("What happens if the learning rate is too large in SFT?")
print("Answer: Catastrophic forgetting — the model forgets pretrained knowledge,")
print("      only memorizes SFT data patterns, losing generalization ability.")

# Exercise 4: Chat Template
print("\nExercise 4: Multi-turn Chat Template")
multi_turn = [
    {"role": "system", "content": "You are a math teacher."},
    {"role": "user", "content": "What is 1+1?"},
    {"role": "assistant", "content": "1+1=2."},
    {"role": "user", "content": "What about 2+2?"},
    {"role": "assistant", "content": "2+2=4."},
]
result = apply_chat_template(multi_turn)
print(f"Formatted result:\n{result}")

print("\n" + "=" * 70)
print("Lesson 21 Summary:")
print("  1. SFT goal: Teach pretrained model to follow instructions")
print("  2. Chat Template: Format conversations into model-readable text")
print("  3. Loss Mask: Only compute loss on assistant responses (ignore_index=-100)")
print("  4. Key tips: Small LR + few epochs + start from pretrained weights")
print("  5. Catastrophic forgetting: Large LR destroys pretrained knowledge")
print("=" * 70)
