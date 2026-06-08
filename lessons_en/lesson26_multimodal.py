# ============================================================================
# Lesson 26: Multimodal (Vision + Language)
# ============================================================================
"""
In the previous 25 lessons we only dealt with text. But the real world has
images, audio, and more. Multimodal models let LLMs "see and describe" —
taking image input and generating text descriptions or answers.

Topics:
1. Why multimodal? Text-only vs multimodal
2. Vision Encoder: How to turn images into vectors (simplified ViT)
3. Vision-Language Projector: How to "translate" visual features for the LLM
4. Multimodal Fusion: How to combine image tokens and text tokens
5. Training a mini multimodal model
6. Inference demo: image captioning

Key Concepts:
- Vision Encoder: Slice image into patches, encode as vector sequence
- Projector: Map visual features into the LLM's embedding space
- Fusion: Concatenate image tokens and text tokens for the LLM
- VLM (Vision-Language Model): Model that understands both images and text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# Global random seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# 1. Why Multimodal?
# ============================================================================
"""
Text-only LLM limitation:
  Input: "What's in this image?" → Model can't answer, it "can't see" images

Multimodal VLM capability:
  Input: [Image] + "What's in this image?"
  Output: "An orange cat sitting on a keyboard"  ← Can see and describe!

Mainstream multimodal architectures:
┌──────────────────────────────────────────────────┐
│  Image → [Vision Encoder] → Visual features      │
│                                    ↓             │
│                          [Projector] → Vision tokens
│                                    ↓             │
│  Text → [Tokenizer] → Text tokens ──→ [LLM] → Output
└──────────────────────────────────────────────────┘

Representative models:
- LLaVA: ViT + MLP Projector + LLaMA
- Qwen-VL: ViT + Cross-Attention + Qwen
- MiniGPT-4: ViT + Q-Former + Vicuna
"""

print("=" * 70)
print("Lesson 26: Multimodal (Vision + Language)")
print("=" * 70)

# ============================================================================
# 2. Vision Encoder — Simplified ViT
# ============================================================================
"""
Vision Transformer (ViT) core idea:
1. Slice image into small patches, treat each patch as a "word"
2. Use linear projection to turn patches into vectors
3. Add positional embeddings
4. Feed into Transformer encoder

Real ViT (e.g., CLIP ViT-L/14):
  - Input: 224x224x3 image
  - Patch size: 14x14 → 16x16 = 256 patches
  - Hidden dim: 1024
  - Layers: 24

Our simplified version:
  - Input: 32x32x3 image (simulated with random data)
  - Patch size: 8x8 → 4x4 = 16 patches
  - Hidden dim: 64
  - Layers: 2
"""


class PatchEmbedding(nn.Module):
    """Slice image into patches and linearly project to vectors

    Equivalent to Conv2d with kernel_size=stride=patch_size,
    but using unfold + Linear is more intuitive for teaching.
    """

    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 4x4 = 16
        self.patch_dim = in_channels * patch_size * patch_size  # 3x8x8 = 192
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) → (B, n_patches, patch_dim)
        B, C, H, W = x.shape

        # [unfold Step-by-Step Breakdown]
        #
        # Original image: (B, 3, 32, 32) — 3 channels, 32x32 pixels
        #
        # Step 1: x.unfold(2, 8, 8) — slice patches along height
        #   On dim 2 (H), slide a window of size 8 with stride 8
        #   Result: (B, 3, 4, 32, 8) — 4 height positions, each 8 rows
        #   Explanation: 32/8=4, so height is split into 4 blocks
        #
        # Step 2: .unfold(3, 8, 8) — slice patches along width
        #   On dim 3 (W), slide a window of size 8 with stride 8
        #   Result: (B, 3, 4, 4, 8, 8) — 4x4=16 patches, each 8x8 pixels
        #   Explanation: 32/8=4, so width is also split into 4 blocks
        #
        # Step 3: view(B, C, -1, 8, 8)
        #   Result: (B, 3, 16, 8, 8) — 16 patches, each 3 channels 8x8
        #
        # Step 4: permute(0, 2, 3, 4, 1).view(B, 16, -1)
        #   Move channel dim to last, then flatten
        #   Result: (B, 16, 3x8x8) = (B, 16, 192) — 16 patches, each a 192-dim vector
        #
        # Analogy: Cut a photo into 16 small photos, flatten each into a vector

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, self.n_patches, -1)
        return self.proj(patches)  # (B, n_patches, embed_dim)


class VisionEncoder(nn.Module):
    """Simplified ViT encoder

    Structure: PatchEmbed → PosEmb → N × TransformerEncoderLayer → Norm
    """

    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64, n_layers=2, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
                dropout=0.0, batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x) + self.pos_embed  # (B, n_patches, embed_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # (B, n_patches, embed_dim)
        return x


# Demo vision encoder
print("\n--- Vision Encoder Example ---")
dummy_image = torch.randn(2, 3, 32, 32)  # 2 images, 32x32 RGB
vision_enc = VisionEncoder(img_size=32, patch_size=8, embed_dim=64, n_layers=2, n_heads=4)
vision_features = vision_enc(dummy_image)
print(f"Input image: {dummy_image.shape}  (B, C, H, W)")
print(f"Visual features: {vision_features.shape}  (B, n_patches, embed_dim)")
print(f"Number of patches: {vision_enc.n_patches}  ({32//8}x{32//8})")

# ============================================================================
# 3. Vision-Language Projector
# ============================================================================
"""
The vision encoder output dimension ≠ LLM embedding dimension.
A Projector is needed to "translate" visual features into the language space.

Common approaches:
1. Linear Projection: Simplest, used in LLaVA v1
   - Pros: Few parameters, fast training
   - Cons: Limited expressiveness

2. MLP Projection (2-layer MLP): Used in LLaVA v1.5
   - Pros: More expressive
   - Cons: Slightly more parameters

3. Q-Former: Used in BLIP-2 / InstructBLIP
   - Pros: Learnable queries, compresses visual tokens
   - Cons: Complex structure

We use MLP projection (best balance for teaching).
"""


class MLPProjector(nn.Module):
    """MLP Projector: Map visual features to LLM embedding space

    Structure: Linear → GELU → Linear
    """

    def __init__(self, vision_dim=64, llm_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        return self.proj(x)  # (B, n_patches, llm_dim)


# Demo projector
print("\n--- Projector Example ---")
projector = MLPProjector(vision_dim=64, llm_dim=64)
vision_tokens = projector(vision_features)
print(f"Visual features: {vision_features.shape}  (vision_dim=64)")
print(f"After projection: {vision_tokens.shape}  (llm_dim=64)")

# ============================================================================
# 4. Multimodal Fusion — Image tokens + Text tokens
# ============================================================================
"""
Fusion methods (simple to complex):

1. Concatenation ← We use this, most intuitive
   [IMG_TOK_1, ..., IMG_TOK_N, TEXT_TOK_1, ..., TEXT_TOK_M] → LLM
   - Simple and direct, used by LLaVA series
   - Image and text tokens concatenated along sequence dimension

2. Cross-Attention
   Text tokens as Q, image tokens as K/V
   - Used by Qwen-VL
   - Text can "query" specific regions of the image

3. Prefix Fusion
   Image tokens placed before text as "prefix"
   - Essentially the same as concatenation, emphasizing image-first order

We use concatenation: [image tokens] + [text tokens] → LLM
"""


class SimpleMultimodalGPT(nn.Module):
    """Mini Multimodal GPT

    Architecture: VisionEncoder → Projector → [vision tokens + text tokens] → CausalLM

    Training:
      Input: [Image] + "<|im_start|>user\nDescribe this image<|im_end|><|im_start|>assistant\n"
      Labels: Only compute loss on assistant parts (same Loss Mask as SFT)

    Inference:
      Input: [Image] + "<|im_start|>user\nDescribe this image<|im_end|><|im_start|>assistant\n"
      Output: "A cat sitting on a keyboard..."
    """

    def __init__(self, vocab_size, img_size=32, patch_size=8, embed_dim=64, n_layers=2, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.head_dim = embed_dim

        # Vision encoder
        self.vision_encoder = VisionEncoder(img_size, patch_size, 3, embed_dim, n_layers=2, n_heads=4)
        self.n_patches = self.vision_encoder.n_patches

        # Projector
        self.projector = MLPProjector(vision_dim=embed_dim, llm_dim=embed_dim)

        # Language model components
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # Causal Transformer
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': CausalSelfAttention(embed_dim, embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': SimpleFFN(embed_dim),
            }))
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight sharing
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, image, text_ids, targets=None):
        """
        Args:
            image: (B, C, H, W) input images
            text_ids: (B, T) text token ids
            targets: (B, T) labels (-100 means no loss)
        """
        B = text_ids.shape[0]

        # 1. Vision encoding
        vision_feat = self.vision_encoder(image)  # (B, n_patches, embed_dim)
        vision_tokens = self.projector(vision_feat)  # (B, n_patches, embed_dim)

        # 2. Text encoding
        T = text_ids.shape[1]
        text_tokens = self.tok_emb(text_ids)  # (B, T, embed_dim)

        # 3. Concatenation: [vision tokens | text tokens]
        combined = torch.cat([vision_tokens, text_tokens], dim=1)  # (B, n_patches+T, embed_dim)

        # 4. Add positional embeddings
        seq_len = combined.shape[1]
        positions = torch.arange(seq_len, device=text_ids.device)
        combined = combined + self.pos_emb(positions)

        # 5. Causal Transformer
        for block in self.blocks:
            combined = combined + block['attn'](block['norm1'](combined))
            combined = combined + block['ffn'](block['norm2'](combined))
        combined = self.norm(combined)

        # 6. Compute logits only for text part (vision tokens don't predict words)
        text_logits = combined[:, self.n_patches:, :]  # (B, T, embed_dim)
        logits = self.lm_head(text_logits)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, image, text_ids, max_new_tokens=20, temperature=1.0, eos_id=None):
        """Multimodal generation: given image and text prompt, generate answer"""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(image, text_ids)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            text_ids = torch.cat([text_ids, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
            if text_ids.shape[1] >= self.max_len - self.n_patches:
                break
        return text_ids

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Reuse causal attention and FFN from Lesson 21
class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention"""

    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, head_dim, bias=False)
        self.k_proj = nn.Linear(dim, head_dim, bias=False)
        self.v_proj = nn.Linear(dim, head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim, dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        return self.o_proj(weights @ v)


class SimpleFFN(nn.Module):
    """Standard FFN: Linear → GELU → Linear"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ============================================================================
# 5. Multimodal Dataset
# ============================================================================


class SimpleMultimodalDataset:
    """Simplified multimodal dataset

    Each sample contains: one image + one text conversation
    Images are simulated with random data, text encoded with character-level tokenizer
    """

    def __init__(self, data, img_size=32, max_length=96):
        self.data = data
        self.img_size = img_size
        self.max_length = max_length

        # Build character-level vocabulary
        all_chars = set()
        for sample in data:
            for key in ("prompt", "answer"):
                all_chars.update(sample[key])
        common_chars = " .,!?;:()\"'" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789"
        special_chars = "<|im_start|>system\nuser\nassistant\n<|im_end|><image>"
        all_chars.update(special_chars)
        all_chars.update(common_chars)
        chars = sorted(all_chars)

        self.pad_token_id = 0
        self.unk_token_id = 1
        special_start = len(chars) + 2
        self.id_im_start = special_start
        self.id_assistant = special_start + 1
        self.id_eos = special_start + 2
        self.id_user = special_start + 3
        self.id_system = special_start + 4
        self.id_image = special_start + 5  # <image> special token

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
        self.id2char[self.id_image] = "<image>"

        self.vocab_size = len(chars) + 2 + 6  # +6 special tokens

    def _encode_text(self, text):
        return [self.char2id.get(c, self.unk_token_id) for c in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Generate simulated image (fixed seed ensures same image per sample)
        torch.manual_seed(idx + 1000)
        image = torch.randn(3, self.img_size, self.img_size)

        # Encode text: [BOS_USER, prompt, EOS, BOS_AST, answer, EOS]
        prompt_ids = self._encode_text(sample["prompt"])
        answer_ids = self._encode_text(sample["answer"])

        text_ids = (
            self.bos_user_ids + prompt_ids + self.eos_ids +
            self.bos_assistant_ids + answer_ids + self.eos_ids
        )

        # Truncate / pad
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
        else:
            text_ids = text_ids + [self.pad_token_id] * (self.max_length - len(text_ids))

        # Construct labels: only compute loss on assistant parts
        labels = self._create_labels(text_ids)

        return image, torch.tensor(text_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _create_labels(self, text_ids):
        """Same Loss Mask logic as Lesson 21"""
        labels = [-100] * len(text_ids)
        bsz = len(self.bos_assistant_ids)
        esz = len(self.eos_ids)
        i = 0
        while i < len(text_ids):
            if i + bsz <= len(text_ids) and text_ids[i:i + bsz] == self.bos_assistant_ids:
                start = i + bsz
                j = start
                while j < len(text_ids) and not (j + esz <= len(text_ids) and text_ids[j:j + esz] == self.eos_ids):
                    j += 1
                for k in range(start, min(j + esz, len(text_ids))):
                    if text_ids[k] != self.pad_token_id:
                        labels[k] = text_ids[k]
                i = j + esz
            else:
                i += 1
        return labels


# Create simulated multimodal data
mm_data = [
    {"prompt": "Describe this image.", "answer": "An orange cat in the picture."},
    {"prompt": "What colors are there?", "answer": "Red and blue."},
    {"prompt": "What scene is this?", "answer": "This is a park scene."},
    {"prompt": "How many people are in the image?", "answer": "There are two people."},
    {"prompt": "How is the weather?", "answer": "Sunny and bright."},
]

mm_dataset = SimpleMultimodalDataset(mm_data)
image, text_ids, labels = mm_dataset[0]
print(f"\n--- Multimodal Dataset Example ---")
print(f"Image: {image.shape}  (C, H, W)")
print(f"Text: {text_ids.shape}  (max_length,)")
print(f"Labels: {labels.shape}  (max_length,)")
print(f"Tokens with loss: {(labels != -100).sum().item()}")
print(f"Vocab size: {mm_dataset.vocab_size}")

# ============================================================================
# 6. Training the Multimodal Model
# ============================================================================


def train_multimodal(model, dataset, epochs=5, lr=1e-3, batch_size=2, seed=42):
    """Multimodal training loop"""
    if seed is not None:
        torch.manual_seed(seed)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Grouped learning rates: smaller LR for vision encoder (pretrained weights are precious)
    vision_params = list(model.vision_encoder.parameters()) + list(model.projector.parameters())
    llm_params = list(model.tok_emb.parameters()) + list(model.lm_head.parameters())
    for block in model.blocks:
        llm_params += list(block.parameters())
    llm_params += list(model.pos_emb.parameters()) + list(model.norm.parameters())

    optimizer = torch.optim.AdamW([
        {"params": vision_params, "lr": lr * 0.1},  # Smaller LR for vision
        {"params": llm_params, "lr": lr},
    ], weight_decay=0.01)

    # Learning rate scheduler
    total_steps = max(len(loader) * epochs, 1)
    warmup_steps = max(int(total_steps * 0.1), 1)

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
        for images, text_ids, labels in loader:
            optimizer.zero_grad()
            logits, loss = model(images, text_ids, targets=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

    return model


# Run training
print("\n--- Multimodal Model Training ---")
torch.manual_seed(42)
mm_model = SimpleMultimodalGPT(
    vocab_size=mm_dataset.vocab_size,
    img_size=32, patch_size=8,
    embed_dim=64, n_layers=2, max_len=128
)
print(f"Model parameters: {mm_model.num_params() / 1e3:.1f}K")

print("\n[Training Start]")
mm_model = train_multimodal(mm_model, mm_dataset, epochs=8, lr=1e-3, batch_size=2, seed=42)

# ============================================================================
# 7. Inference Demo
# ============================================================================
print("\n--- Multimodal Inference Demo ---")
mm_model.eval()

# Simulate a new image
test_image = torch.randn(1, 3, 32, 32)
test_prompt = "Describe this image."
prompt_ids = mm_dataset._encode_text(test_prompt)
start_ids = mm_dataset.bos_user_ids + prompt_ids + mm_dataset.eos_ids + mm_dataset.bos_assistant_ids
text_input = torch.tensor([start_ids], dtype=torch.long)

generated = mm_model.generate(test_image, text_input, max_new_tokens=15, temperature=0.8, eos_id=mm_dataset.id_eos)
decoded = "".join(mm_dataset.id2char.get(int(t), '?') for t in generated[0])
print(f"Prompt: {test_prompt}")
print(f"Generated: {decoded}")

# ============================================================================
# 8. Key Multimodal Training Tips
# ============================================================================
"""
1. Vision encoder is usually frozen
   - Pretrained ViT is already strong, no need to retrain
   - Only train Projector + LLM
   - Saves memory and training time

2. Staged training
   - Stage 1: Freeze ViT + LLM, only train Projector (align visual-language spaces)
   - Stage 2: Unfreeze LLM, joint fine-tuning (improve multimodal understanding)
   - Stage 3: Unfreeze all, small LR fine-tuning (optional)

3. Grouped learning rates
   - ViT: Smallest LR (1e-6), pretrained weights are most precious
   - Projector: Medium LR (1e-4), needs to align quickly
   - LLM: Larger LR (1e-5), needs to adapt to multimodal input

4. Image resolution
   - Use low resolution (224x224) during training to save memory
   - Can use high resolution at inference for better results
   - Any resolution must be a multiple of patch_size

5. Data quality
   - Image-text alignment is key: image and description must match
   - Data quantity doesn't need to be huge, but quality matters
   - Common datasets: COCO Captions, LAION, ShareGPT4V
"""

# ============================================================================
# 9. Freezing Vision Encoder Experiment
# ============================================================================
print("\n--- Freezing Vision Encoder Experiment ---")

torch.manual_seed(42)
frozen_model = SimpleMultimodalGPT(
    vocab_size=mm_dataset.vocab_size,
    img_size=32, patch_size=8,
    embed_dim=64, n_layers=2, max_len=128
)

# Freeze vision encoder
for param in frozen_model.vision_encoder.parameters():
    param.requires_grad = False

# Count trainable parameters
total_params = frozen_model.num_params()
trainable_params = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params / 1e3:.1f}K")
print(f"Trainable parameters: {trainable_params / 1e3:.1f}K ({trainable_params / total_params * 100:.0f}%)")
print(f"Frozen parameters: {(total_params - trainable_params) / 1e3:.1f}K ({(1 - trainable_params / total_params) * 100:.0f}%)")

print("\n[Frozen ViT Training]")
frozen_model = train_multimodal(frozen_model, mm_dataset, epochs=5, lr=1e-3, batch_size=2, seed=42)

# ============================================================================
# 10. Multimodal vs Text-Only Comparison
# ============================================================================
print("\n--- Multimodal vs Text-Only Model Comparison ---")


class TextOnlyGPT(nn.Module):
    """Text-only GPT (for comparison)"""

    def __init__(self, vocab_size, embed_dim=64, n_layers=2, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.head_dim = embed_dim

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': CausalSelfAttention(embed_dim, embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': SimpleFFN(embed_dim),
            }))
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
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
            x = x + block['attn'](block['norm1'](x))
            x = x + block['ffn'](block['norm2'](x))
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


torch.manual_seed(42)
text_model = TextOnlyGPT(vocab_size=mm_dataset.vocab_size, embed_dim=64, n_layers=2)

print(f"Multimodal model parameters: {mm_model.num_params() / 1e3:.1f}K")
print(f"Text-only model parameters: {text_model.num_params() / 1e3:.1f}K")
print(f"Vision encoder share: {sum(p.numel() for p in mm_model.vision_encoder.parameters()) / mm_model.num_params() * 100:.0f}%")

# ============================================================================
# Exercises
# ============================================================================
print("\n" + "=" * 70)
print("Exercises (think/practice first, then check reference answers below)")
print("=" * 70)

# Exercise 1: Modify fusion method
print("\nExercise 1: Modify fusion method")
print("  Current: [vision tokens] + [text tokens] concatenation")
print("  Change to: [text tokens] + [vision tokens] + [text tokens] three-way concat")
print("  i.e.: prompt text → image → answer text")
print("  Hint: Modify the torch.cat order in SimpleMultimodalGPT.forward")
print("  Think: What are the advantages of this order? How does it differ from LLaVA?")

# Exercise 2: Freeze different components
print("\nExercise 2: Freeze different components")
print("  Current: Vision encoder is frozen")
print("  Try:")
print("  (a) Freeze LLM, only train Projector + ViT")
print("  (b) Freeze everything, only train Projector")
print("  (c) Freeze nothing")
print("  Compare training loss and generation quality across all four strategies.")

# Exercise 3: Add image augmentation
print("\nExercise 3: Image augmentation")
print("  Current: Using random noise to simulate images")
print("  Add simple image augmentation in __getitem__:")
print("  - Random horizontal flip")
print("  - Random crop")
print("  - Color jitter")
print("  Hint: Use torchvision.transforms or implement manually")

# Exercise 4: Thinking question
print("\nExercise 4: Why does the vision encoder use bidirectional attention, while the language model uses causal attention?")
print("  ViT's TransformerEncoderLayer is bidirectional (can see all patches),")
print("  while GPT's CausalSelfAttention is causal (can only see previous tokens).")
print("  Why this design? What if ViT also used causal attention?")

# Exercise 4 reference answer
print("\n[Reference Answer · Exercise 4]")
print("  Images are complete — all patches exist simultaneously with no temporal order,")
print("  so ViT using bidirectional attention to let each patch see the whole image is reasonable.")
print("  Text is generated token by token — the current token cannot 'peek' at future tokens,")
print("  so LLMs must use causal attention.")
print("  If ViT used causal attention, each patch could only see patches above and to the left,")
print("  severely damaging visual understanding (bottom-right objects would 'not see' top-left context).")

print("\n" + "=" * 70)
print("Lesson 26 Summary:")
print("  1. Multimodal model = Vision Encoder + Projector + Language Model")
print("  2. ViT slices images into patches, encodes as vector sequences")
print("  3. Projector maps visual features to LLM embedding space")
print("  4. Fusion methods: Concatenation (simplest), Cross-attention (more flexible)")
print("  5. Training tips: Freeze ViT, staged training, grouped learning rates")
print("  6. Vision encoder uses bidirectional attention, language model uses causal attention")
print("=" * 70)
