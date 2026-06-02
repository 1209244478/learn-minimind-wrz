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
import random

# Global random seed (ensures all experiments in this lesson are reproducible)
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

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


def apply_chat_template(messages):
    """Simple ChatML template implementation

    Args:
        messages: Conversation list, e.g. [{"role": "user", "content": "Hello"}, ...]

    Returns:
        Formatted conversation text
    """
    # ChatML format (Qwen/MiniMind): <|im_start|>role\ncontent<|im_end|>\n
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

    Uses single-pass + state machine implementation, O(N) complexity.
    Once inside an assistant span, mark all tokens until EOS;
    padding positions remain -100 (not included in loss).

    Args:
        input_ids: Token ID list for the full conversation
        bos_assistant_ids: Token IDs for "<|im_start|>assistant\n"
        eos_ids: Token IDs for "<|im_end|>"
        pad_token_id: Padding token ID

    Returns:
        labels: Same length as input_ids, non-assistant parts set to -100
    """
    n = len(input_ids)
    labels = [-100] * n
    bsz = len(bos_assistant_ids)
    esz = len(eos_ids)
    i = 0

    while i < n:
        # Match assistant start marker: enter assistant span
        if i + bsz <= n and input_ids[i:i + bsz] == bos_assistant_ids:
            start = i + bsz
            j = start
            # Find EOS within the span
            while j < n and not (j + esz <= n and input_ids[j:j + esz] == eos_ids):
                j += 1
            # Mark [start, j + esz) as valid (EOS itself should also be learned)
            for k in range(start, min(j + esz, n)):
                if input_ids[k] != pad_token_id:
                    labels[k] = input_ids[k]
            i = j + esz
        else:
            i += 1

    return labels


# Demo label construction
print("\n--- SFT Label Construction Example ---")
# Simulate a simple token sequence
# Assume: [BOS_SYS] system_content [EOS] [BOS_USR] user_content [EOS] [BOS_AST] answer [EOS] [PAD]...
simple_input = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 0, 0]
# BOS_AST = [30], EOS = [34]
labels = create_sft_labels(simple_input, bos_assistant_ids=[30], eos_ids=[34], pad_token_id=0)

print(f"input_ids: {simple_input}")
print(f"labels:    {labels}")
print(f"Positions with loss: {[i for i, l in enumerate(labels) if l != -100]}")

# ============================================================================
# 4. SFT Dataset
# ============================================================================


class SimpleSFTDataset:
    """SFT Dataset

    Uses character-level tokenizer to encode conversation text,
    constructing input_ids and labels (only assistant parts compute loss).

    Key fix: Use different BOS markers for user and assistant segments,
    so create_sft_labels only matches assistant segments,
    preventing user prompts from being incorrectly included in loss.
    """

    def __init__(self, data, max_length=64, shared_vocab=None):
        self.data = data
        self.max_length = max_length

        if shared_vocab is not None:
            # Reuse existing vocabulary (validation set must share training set vocab)
            self.char2id = shared_vocab["char2id"].copy()
            self.id2char = shared_vocab["id2char"].copy()
            self.vocab_size = shared_vocab["vocab_size"]
            self.pad_token_id = shared_vocab["pad_token_id"]
            self.unk_token_id = shared_vocab["unk_token_id"]
            self.id_im_start = shared_vocab["id_im_start"]
            self.id_assistant = shared_vocab["id_assistant"]
            self.id_eos = shared_vocab["id_eos"]
            self.id_user = shared_vocab["id_user"]
            self.id_system = shared_vocab["id_system"]
            self.bos_user_ids = shared_vocab["bos_user_ids"]
            self.bos_assistant_ids = shared_vocab["bos_assistant_ids"]
            self.eos_ids = shared_vocab["eos_ids"]
            return

        # Build character-level vocabulary: training data + ChatML special chars + common chars
        all_chars = set()
        for sample in data:
            for key in ("content", "answer"):
                all_chars.update(sample[key])
        # Add ChatML special characters and common English/Chinese characters to avoid UNK
        # (demo dataset is small; adding common chars ensures new inputs are encoded correctly)
        common_chars = " .,!?;:()\"'" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789"
        special_chars = "<|im_start|>system\nuser\nassistant\n<|im_end|>"
        all_chars.update(special_chars)
        all_chars.update(common_chars)
        chars = sorted(all_chars)

        self.pad_token_id = 0
        self.unk_token_id = 1
        # Special token ID allocation (after regular characters):
        #   0      -> <pad>
        #   1      -> <unk>
        #   [2, 2+len(chars)) -> characters
        #   +0     -> <|im_start|>       (shared token)
        #   +1     -> assistant\n
        #   +2     -> <|im_end|>
        #   +3     -> user\n
        #   +4     -> system\n
        special_start = len(chars) + 2
        self.id_im_start = special_start
        self.id_assistant = special_start + 1
        self.id_eos = special_start + 2
        self.id_user = special_start + 3
        self.id_system = special_start + 4

        # Key: Use id_user for user segment, id_assistant for assistant segment
        # This way create_sft_labels only matches assistant segments
        self.bos_user_ids = [self.id_im_start, self.id_user]
        self.bos_assistant_ids = [self.id_im_start, self.id_assistant]
        self.eos_ids = [self.id_eos]

        self.char2id = {c: i + 2 for i, c in enumerate(chars)}
        self.id2char = {i + 2: c for i, c in enumerate(chars)}
        self.id2char[self.pad_token_id] = "<pad>"
        self.id2char[self.unk_token_id] = "<unk>"
        self.id2char[self.id_im_start] = "<|im_start|>"
        self.id2char[self.id_assistant] = "assistant\n"
        self.id2char[self.id_user] = "user\n"
        self.id2char[self.id_system] = "system\n"
        self.id2char[self.id_eos] = "<|im_end|>"

        # Vocab size = pad + unk + regular chars + 5 special tokens
        self.vocab_size = len(chars) + 2 + 5

    def get_vocab(self):
        """Export vocabulary info for validation/test set reuse"""
        return {
            "char2id": self.char2id,
            "id2char": self.id2char,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "id_im_start": self.id_im_start,
            "id_assistant": self.id_assistant,
            "id_eos": self.id_eos,
            "id_user": self.id_user,
            "id_system": self.id_system,
            "bos_user_ids": self.bos_user_ids,
            "bos_assistant_ids": self.bos_assistant_ids,
            "eos_ids": self.eos_ids,
        }

    def _encode_text(self, text):
        """Encode text using character-level tokenizer"""
        return [self.char2id.get(c, self.unk_token_id) for c in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return (input_ids, labels) pair

        Sequence structure:
          [BOS_USER, prompt_tokens, EOS, BOS_AST, answer_tokens, EOS, PAD...]
        """
        sample = self.data[idx]

        prompt_ids = self._encode_text(sample["content"])
        answer_ids = self._encode_text(sample["answer"])

        # Concatenate: user segment + assistant segment (with different BOS markers)
        input_ids = (
            self.bos_user_ids + prompt_ids + self.eos_ids +
            self.bos_assistant_ids + answer_ids + self.eos_ids
        )

        # Truncate or pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids = input_ids + [self.pad_token_id] * (self.max_length - len(input_ids))

        # Construct labels: only compute loss on assistant response parts
        labels = create_sft_labels(input_ids, self.bos_assistant_ids, self.eos_ids, self.pad_token_id)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# Create simulated SFT data
sft_data = [
    {"role": "user", "content": "What is machine learning?", "answer": "ML is a branch of AI..."},
    {"role": "user", "content": "What is Python?", "answer": "Python is a programming language..."},
    {"role": "user", "content": "Deep learning vs machine learning?", "answer": "DL is a subset of ML..."},
    {"role": "user", "content": "Hello, please introduce yourself.", "answer": "Hello! I am MiniMind, a mini language model."},
    {"role": "user", "content": "What is 1+1?", "answer": "1+1=2."},
]

# Reserve 1 sample for validation, use 4 for training
sft_train_data = sft_data[:4]
sft_val_data = sft_data[4:]

sft_dataset = SimpleSFTDataset(sft_train_data)
sft_val_dataset = SimpleSFTDataset(sft_val_data, max_length=sft_dataset.max_length,
                                   shared_vocab=sft_dataset.get_vocab()) if sft_val_data else None

input_ids, labels = sft_dataset[0]
print(f"\n--- SFT Dataset Example ---")
print(f"input_ids shape: {input_ids.shape}")
print(f"labels shape: {labels.shape}")
print(f"Tokens with loss: {(labels != -100).sum().item()}")
print(f"Total tokens: {labels.shape[0]}")
# Decode first 20 tokens back to characters for visual inspection
print(f"Decoded first 20 tokens: {''.join(sft_dataset.id2char.get(int(t), '?') for t in input_ids[:20])}")

# ============================================================================
# 5. SFT Training Loop
# ============================================================================


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention (simplified for teaching)

    Key point: Upper-triangular mask ensures token i can only see tokens [0, i],
    which is the fundamental difference between GPT-style and bidirectional Transformers.
    """

    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, head_dim, bias=False)
        self.k_proj = nn.Linear(dim, head_dim, bias=False)
        self.v_proj = nn.Linear(dim, head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim, dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x)  # (B, T, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Attention scores + scaling
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, T, T)
        # Causal mask: upper triangle (above diagonal) set to -inf
        # Note: Teaching version rebuilds mask each time; production code should pre-compute and cache
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # (B, T, head_dim)
        return self.o_proj(out)


class SimpleFFN(nn.Module):
    """Standard FFN: Linear -> GELU -> Linear (teaching version, uses GELU for simplicity)"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SimpleBlock(nn.Module):
    """Pre-norm Transformer Block: Attention + FFN"""

    def __init__(self, dim, head_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SimpleFFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleGPT(nn.Module):
    """Simplified GPT (for demo)

    Difference from original (Fix #2):
      - No longer uses nn.TransformerEncoderLayer (bidirectional attention)
      - Uses hand-written causal self-attention + Pre-norm + weight sharing
      - Can actually be used as a next-token-prediction LM
    """

    def __init__(self, vocab_size, dim=64, n_layers=2, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        # Single-head attention (head_dim == dim), fewer params for demo
        self.head_dim = dim

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([SimpleBlock(dim, self.head_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Weight sharing: Embedding and LM Head share parameters
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.tok_emb(ids)
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
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=20, temperature=1.0, eos_id=None):
        """Simple autoregressive generation (greedy, truncated to max_len)"""
        self.eval()
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.max_len:]
            logits, _ = self(ids_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return ids

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_sft(model, dataset, epochs=3, lr=1e-5, batch_size=4,
              val_dataset=None, warmup_ratio=0.1, max_grad_norm=1.0,
              log_interval=1, seed=None, verbose=True):
    """SFT Training Loop

    Key differences from pretraining:
    1. Smaller learning rate (1e-5 vs 1e-4)
    2. Fewer epochs (1-3 vs 3-10)
    3. Loss only on assistant parts
    4. Start from pretrained weights (not random init)

    Args:
        model: Model to train
        dataset: Training set
        epochs: Number of training epochs
        lr: Learning rate (SFT typically uses 1e-5 ~ 5e-5)
        batch_size: Batch size
        val_dataset: Validation set (optional; if provided, val loss computed every log_interval epochs)
        warmup_ratio: Fraction of total steps for linear LR warmup
        max_grad_norm: Gradient clipping threshold
        log_interval: Print log every N epochs
        seed: Random seed (for DataLoader and model init reproducibility)
        verbose: Whether to print logs
    """
    if seed is not None:
        torch.manual_seed(seed)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=len(dataset) >= batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler: linear warmup + cosine decay (standard SFT config)
    total_steps = max(len(loader) * epochs, 1)
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for input_ids, labels in loader:
            optimizer.zero_grad()
            logits, loss = model(input_ids, targets=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and ((epoch + 1) % log_interval == 0 or epoch == epochs - 1):
            avg_loss = total_loss / max(n_batches, 1)
            current_lr = scheduler.get_last_lr()[0]
            log = f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
            if val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    val_total, val_n = 0.0, 0
                    for vid, vlab in val_dataset:
                        _, vl = model(vid.unsqueeze(0), targets=vlab.unsqueeze(0))
                        val_total += vl.item()
                        val_n += 1
                    log += f", Val Loss: {val_total / max(val_n, 1):.4f}"
                model.train()
            print(log)

    return model


# Run SFT training
print("\n--- SFT Training Demo ---")
torch.manual_seed(42)
model = SimpleGPT(vocab_size=sft_dataset.vocab_size, dim=64, n_layers=2)
print(f"Model parameters: {model.num_params() / 1e3:.1f}K")
print(f"Vocab size: {sft_dataset.vocab_size}")

# Simulate: first "pretrain" init, then do SFT
print("\n[SFT Training Start]")
model = train_sft(model, sft_dataset, epochs=8, lr=1e-3, batch_size=2,
                  val_dataset=sft_val_dataset, warmup_ratio=0.1, seed=42)

# ============================================================================
# 6. SFT vs Pretrain Loss Comparison
# ============================================================================
"""
Let's compare the difference in loss computation between SFT and pretraining:
"""

print("\n--- SFT vs Pretrain Loss Comparison ---")

# Simulate a batch of data
torch.manual_seed(42)
B, T, V = 2, 16, 100
logits = torch.randn(B, T, V)

# Method 1: Pretrain — all tokens compute loss
pretrain_labels = torch.randint(0, V, (B, T))
pretrain_loss = F.cross_entropy(logits.view(-1, V), pretrain_labels.view(-1))
print(f"Pretrain loss (all tokens): {pretrain_loss.item():.4f}")

# Method 2: SFT — only assistant parts compute loss
sft_labels = pretrain_labels.clone()
# Assume first 10 tokens are prompt, set to -100
sft_labels[:, :10] = -100
sft_loss = F.cross_entropy(logits.view(-1, V), sft_labels.view(-1), ignore_index=-100)
n_valid = (sft_labels != -100).sum().item()
n_total = sft_labels.numel()
print(f"SFT loss (assistant only): {sft_loss.item():.4f}")
print(f"Tokens with loss: {n_valid}/{n_total} ({n_valid / n_total * 100:.0f}%)")

# ============================================================================
# 6.5 Loss Mask Effect Comparison: Training vs Generation
# ============================================================================
"""
Fix #8: Visually demonstrate the training effect of "with/without Loss Mask"
using generation results.

Method: Train two control models, have them continue from the same prompt.
  - Model A: Trained with SFT labels (only assistant parts compute loss) — correct
  - Model B: Trained with full labels (all tokens including user prompt compute loss) — wrong
Expected: Model A learns to "answer instructions"; Model B learns to "parrot the question then ramble".
"""


class _AllLabelsDataset(SimpleSFTDataset):
    """Control dataset: labels == input_ids, every token computes loss"""
    def __getitem__(self, idx):
        ids, _ = super().__getitem__(idx)
        return ids, ids.clone()


print("\n--- Loss Mask Effect Comparison (Generative) ---")
unmasked_dataset = _AllLabelsDataset(sft_train_data, max_length=sft_dataset.max_length)
unmasked_val = _AllLabelsDataset(sft_val_data, max_length=sft_dataset.max_length,
                                 shared_vocab=sft_dataset.get_vocab()) if sft_val_data else None

torch.manual_seed(42)
model_unmasked = SimpleGPT(vocab_size=sft_dataset.vocab_size, dim=64, n_layers=2)
print("[Control group training: all tokens compute loss]")
model_unmasked = train_sft(model_unmasked, unmasked_dataset, epochs=8, lr=1e-3,
                           batch_size=2, val_dataset=unmasked_val, warmup_ratio=0.1,
                           seed=42, verbose=True)

test_prompt = "What is machine learning?"
prompt_ids = sft_dataset._encode_text(test_prompt)
# Use [BOS_USER, prompt..., EOS, BOS_AST] as starting point, let model continue generating
start_ids = (sft_dataset.bos_user_ids + prompt_ids + sft_dataset.eos_ids + sft_dataset.bos_assistant_ids)
start_tensor = torch.tensor([start_ids], dtype=torch.long)


def gen_text(model, ids, max_new=20):
    out = model.generate(ids, max_new_tokens=max_new, temperature=0.8,
                         eos_id=sft_dataset.id_eos)
    return "".join(sft_dataset.id2char.get(int(t), '?') for t in out[0])


print(f"\nPrompt: {test_prompt}")
print(f"  Correct (SFT Mask):  {gen_text(model, start_tensor.clone())}")
print(f"  Wrong (No Mask):     {gen_text(model_unmasked, start_tensor.clone())}")

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
# 8. Complete SFT Pipeline
# ============================================================================
"""
Complete SFT workflow:

1. Data Preparation
   - Collect instruction-response pair data
   - Format into ChatML conversation format
   - Split into train/validation sets

2. Data Preprocessing
   - Encode conversations with tokenizer
   - Construct labels (only assistant parts compute loss)
   - Padding and truncation

3. Training
   - Load pretrained weights
   - Small learning rate (1e-5) + AdamW
   - Cosine annealing LR schedule
   - Gradient clipping (1.0)
   - Mixed precision training (bf16/fp16)

4. Evaluation
   - Compute loss on validation set
   - Manual evaluation of generation quality
   - Automatic evaluation using GPT-4 etc.

5. Saving
   - Save SFT weights
   - Can be used for subsequent DPO/RLHF training
"""

# ============================================================================
# Exercises
# ============================================================================
print("\n" + "=" * 70)
print("Exercises (think/practice first, then check reference answers below)")
print("=" * 70)

# Exercise 1 (thinking, no complete answer given): Construct SFT labels
print("\nExercise 1: Construct SFT labels")
print("Given the following token sequence:")
test_input = [1, 5, 6, 7, 50, 51, 30, 31, 32, 2, 0, 0]
print(f"  input_ids: {test_input}")
print(f"  BOS_AST = [50, 51], EOS = [2], PAD = 0")
print("Write out the labels by hand (hint: how many tokens in the assistant span? How to handle padding?)")
# Leave a verification function, don't print the answer directly
def _check_ex1(student_labels):
    expected = create_sft_labels(test_input, [50, 51], [2], 0)
    return student_labels == expected, expected
print("  Verification: assign your answer to my_labels, call _check_ex1(my_labels)")
print("  Example: ok, exp = _check_ex1([-100]*12)  # fill in your answer in the list")

# Exercise 2 (hands-on): Change learning rate and retrain, observe Loss curves
print("\nExercise 2: Impact of learning rate on SFT")
print("Modify the lr / epochs below, retrain and compare Loss curves:")
print("  Hint: try lr = 1e-1, 1e-3, 1e-5 each, see which converges and which diverges.")
print("  Code template:")
print("    for lr_ in [1e-1, 1e-3, 1e-5]:")
print("        torch.manual_seed(42)")
print("        m = SimpleGPT(vocab_size=sft_dataset.vocab_size, dim=64, n_layers=2)")
print("        train_sft(m, sft_dataset, epochs=5, lr=lr_, batch_size=2, seed=42)")
print("  Think: which learning rate triggers catastrophic forgetting? Which doesn't train at all?")

# Exercise 3 (hands-on): Construct multi-turn ChatML
print("\nExercise 3: Feed multi-turn conversations to SFT dataset")
print("  Current state: sft_data has only one user+assistant turn per entry.")
print("  Rewrite __getitem__ so input_ids contains 2 turns of user/assistant (multi-turn),")
print("  and ensure create_sft_labels computes loss for both assistant segments.")
print("  Hint: alternate bos_user_ids / eos_ids / bos_assistant_ids.")

# Exercise 4 (thinking): Why use causal mask
print("\nExercise 4: Why must SimpleGPT use a causal mask?")
print("  If you remove the mask in CausalSelfAttention.forward (making it full attention),")
print("  what happens during SFT training? What about training loss and generation results?")
print("  Hint: think about token 0 already 'seeing' token 5's answer,")
print("       but the label tells it 'after token 0 should be X' — what would it learn?")

# Demo code: Exercise 4 answer experiment
print("\n[Reference Answer · Exercise 4 Experiment]")
class BiasedAttention(CausalSelfAttention):
    """Bidirectional attention without mask (for comparison experiment)"""
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # Deliberately no causal mask — this is the 'wrong' version
        weights = F.softmax(scores, dim=-1)
        return self.o_proj(weights @ v)


class BiasedBlock(SimpleBlock):
    def __init__(self, dim, head_dim):
        super().__init__(dim, head_dim)
        self.attn = BiasedAttention(dim, head_dim)


class BiasedGPT(SimpleGPT):
    def __init__(self, vocab_size, dim=64, n_layers=2, max_len=64):
        super().__init__(vocab_size, dim, n_layers, max_len)
        self.blocks = nn.ModuleList([BiasedBlock(dim, self.head_dim) for _ in range(n_layers)])


torch.manual_seed(42)
biased = BiasedGPT(vocab_size=sft_dataset.vocab_size, dim=64, n_layers=2)
train_sft(biased, sft_dataset, epochs=3, lr=1e-3, batch_size=2, seed=42, verbose=False,
          val_dataset=sft_val_dataset)
print("  Bidirectional attention SFT training complete (note val loss difference from normal model)")

# Exercise 4 answer: multi-turn ChatML example
print("\n[Reference Answer · Exercise 3 Example]")
multi_turn = [
    {"role": "system", "content": "You are a math teacher."},
    {"role": "user", "content": "What is 1+1?"},
    {"role": "assistant", "content": "1+1=2."},
    {"role": "user", "content": "What about 2+2?"},
    {"role": "assistant", "content": "2+2=4."},
]
result = apply_chat_template(multi_turn)
print(f"  Formatted result (multi-turn ChatML):\n  {result.replace(chr(10), chr(10) + '  ')}")

print("\n" + "=" * 70)
print("Lesson 21 Summary:")
print("  1. SFT goal: Teach pretrained model to follow instructions")
print("  2. Chat Template: Format conversations into model-readable text")
print("  3. Loss Mask: Only compute loss on assistant responses (ignore_index=-100)")
print("  4. Key tips: Small LR + few epochs + start from pretrained weights")
print("  5. Catastrophic forgetting: Large LR destroys pretrained knowledge")
print("=" * 70)
