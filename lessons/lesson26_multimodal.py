# ============================================================================
# 第26课：多模态 (Multimodal — Vision + Language)
# ============================================================================
"""
前25课我们只处理了文本。但真实世界不仅有文字，还有图像、声音……
多模态模型让 LLM "看图说话"——接收图像输入，生成文本描述或回答。

本课内容：
1. 为什么需要多模态？纯文本 vs 多模态
2. 视觉编码器：如何把图像变成向量？(简化版 ViT)
3. 视觉-语言投影器：如何把视觉特征"翻译"给语言模型？
4. 多模态融合：图像 token 和文本 token 如何拼接？
5. 训练一个迷你多模态模型
6. 推理演示：图像描述生成

关键概念：
- 视觉编码器 (Vision Encoder)：把图像切成 patch，编码为向量序列
- 投影器 (Projector)：将视觉特征映射到语言模型的嵌入空间
- 多模态融合 (Fusion)：图像 token 与文本 token 拼接后一起输入 LLM
- 视觉-语言模型 (VLM)：能同时理解图像和文本的模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# 全局随机种子
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# 1. 为什么需要多模态？
# ============================================================================
"""
纯文本 LLM 的局限：
  输入: "这张图片里有什么？" → 模型无法回答，因为它"看不见"图片

多模态 VLM 的能力：
  输入: [图片] + "这张图片里有什么？"
  输出: "图片中有一只橘猫趴在键盘上"  ← 能看图回答！

主流多模态架构：
┌──────────────────────────────────────────────────┐
│  图像 → [Vision Encoder] → 视觉特征              │
│                                    ↓             │
│                          [Projector] → 视觉token  │
│                                    ↓             │
│  文本 → [Tokenizer] → 文本token ──→ [LLM] → 输出 │
└──────────────────────────────────────────────────┘

代表模型：
- LLaVA: ViT + MLP Projector + LLaMA
- Qwen-VL: ViT + Cross-Attention + Qwen
- MiniGPT-4: ViT + Q-Former + Vicuna
"""

print("=" * 70)
print("第26课：多模态 (Vision + Language)")
print("=" * 70)

# ============================================================================
# 2. 视觉编码器 — 简化版 ViT
# ============================================================================
"""
Vision Transformer (ViT) 的核心思想：
1. 把图像切成小方块 (patch)，每个 patch 看作一个"词"
2. 用线性映射把 patch 变成向量
3. 加上位置编码
4. 送入 Transformer 编码器

真实 ViT (如 CLIP ViT-L/14)：
  - 输入: 224×224×3 图像
  - Patch 大小: 14×14 → 16×16 = 256 个 patch
  - 隐藏维度: 1024
  - 层数: 24

我们的简化版：
  - 输入: 32×32×3 图像（模拟，用随机数据）
  - Patch 大小: 8×8 → 4×4 = 16 个 patch
  - 隐藏维度: 64
  - 层数: 2
"""


class PatchEmbedding(nn.Module):
    """将图像切成 patch 并线性映射为向量

    等价于用 kernel_size=stride=patch_size 的 Conv2d，
    但教学版用 unfold + Linear 更直观。
    """

    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 4×4 = 16
        self.patch_dim = in_channels * patch_size * patch_size  # 3×8×8 = 192
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) → (B, n_patches, patch_dim)
        B, C, H, W = x.shape

        # 【unfold 逐步拆解】
        #
        # 原始图像: (B, 3, 32, 32) — 3通道，32×32像素
        #
        # 第1步：x.unfold(2, 8, 8) — 沿高度方向切 patch
        #   在第2维(H)上，用大小8、步长8的窗口滑动
        #   结果: (B, 3, 4, 32, 8) — 4个高度位置，每个8行
        #   解释：32/8=4，所以高度方向切成4块
        #
        # 第2步：.unfold(3, 8, 8) — 沿宽度方向切 patch
        #   在第3维(W)上，用大小8、步长8的窗口滑动
        #   结果: (B, 3, 4, 4, 8, 8) — 4×4=16个patch，每个8×8像素
        #   解释：32/8=4，所以宽度方向也切成4块
        #
        # 第3步：view(B, C, -1, 8, 8)
        #   结果: (B, 3, 16, 8, 8) — 16个patch，每个3通道8×8
        #
        # 第4步：permute(0, 2, 3, 4, 1).view(B, 16, -1)
        #   把通道维移到最后，然后展平
        #   结果: (B, 16, 3×8×8) = (B, 16, 192) — 16个patch，每个192维向量
        #
        # 类比：把一张照片剪成16张小照片，每张小照片展平成一个向量

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (B, C, H/ps, W/ps, ps, ps) → (B, n_patches, patch_dim)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, self.n_patches, -1)
        return self.proj(patches)  # (B, n_patches, embed_dim)


class VisionEncoder(nn.Module):
    """简化版 ViT 编码器

    结构：PatchEmbed → PosEmb → N × TransformerEncoderLayer → Norm
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


# 演示视觉编码器
print("\n--- 视觉编码器示例 ---")
dummy_image = torch.randn(2, 3, 32, 32)  # 2张 32×32 RGB 图像
vision_enc = VisionEncoder(img_size=32, patch_size=8, embed_dim=64, n_layers=2, n_heads=4)
vision_features = vision_enc(dummy_image)
print(f"输入图像: {dummy_image.shape}  (B, C, H, W)")
print(f"视觉特征: {vision_features.shape}  (B, n_patches, embed_dim)")
print(f"Patch 数量: {vision_enc.n_patches}  ({32//8}×{32//8})")

# ============================================================================
# 3. 视觉-语言投影器
# ============================================================================
"""
视觉编码器输出的特征维度 ≠ 语言模型的嵌入维度，
需要一个投影器 (Projector) 把视觉特征"翻译"到语言空间。

常见方案：
1. 线性投影 (Linear): 最简单，LLaVA v1 使用
   - 优点: 参数少，训练快
   - 缺点: 表达能力有限

2. MLP 投影 (2层 MLP): LLaVA v1.5 使用
   - 优点: 表达能力更强
   - 缺点: 参数稍多

3. Q-Former: BLIP-2 / InstructBLIP 使用
   - 优点: 可学习查询，压缩视觉 token
   - 缺点: 结构复杂

我们用 MLP 投影器（教学最佳平衡点）。
"""


class MLPProjector(nn.Module):
    """MLP 投影器：将视觉特征映射到语言模型的嵌入空间

    结构：Linear → GELU → Linear
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


# 演示投影器
print("\n--- 投影器示例 ---")
projector = MLPProjector(vision_dim=64, llm_dim=64)
vision_tokens = projector(vision_features)
print(f"视觉特征: {vision_features.shape}  (vision_dim=64)")
print(f"投影后:   {vision_tokens.shape}  (llm_dim=64)")

# ============================================================================
# 4. 多模态融合 — 图像 token 与文本 token 拼接
# ============================================================================
"""
融合方式（从简单到复杂）：

1. 拼接融合 (Concatenation) ← 我们用这个，最直观
   [IMG_TOK_1, ..., IMG_TOK_N, TEXT_TOK_1, ..., TEXT_TOK_M] → LLM
   - 简单直接，LLaVA 系列使用
   - 图像 token 和文本 token 在序列维度拼接

2. 交叉注意力 (Cross-Attention)
   文本 token 做 Q，图像 token 做 K/V
   - Qwen-VL 使用
   - 文本可以"查询"图像的特定区域

3. 前缀融合 (Prefix)
   图像 token 放在文本前面作为"前缀"
   - 本质上和拼接一样，只是强调图像在前

我们用拼接融合：[图像token] + [文本token] → LLM
"""


class SimpleMultimodalGPT(nn.Module):
    """迷你多模态 GPT

    架构：VisionEncoder → Projector → [视觉token + 文本token] → CausalLM

    训练时：
      输入: [图像] + "<|im_start|>user\n描述这张图<|im_end|><|im_start|>assistant\n"
      标签: 只对 assistant 部分计算损失（和 SFT 一样用 Loss Mask）

    推理时：
      输入: [图像] + "<|im_start|>user\n描述这张图<|im_end|><|im_start|>assistant\n"
      输出: "图片中有一只猫..."
    """

    def __init__(self, vocab_size, img_size=32, patch_size=8, embed_dim=64, n_layers=2, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.head_dim = embed_dim

        # 视觉编码器
        self.vision_encoder = VisionEncoder(img_size, patch_size, 3, embed_dim, n_layers=2, n_heads=4)
        self.n_patches = self.vision_encoder.n_patches

        # 投影器
        self.projector = MLPProjector(vision_dim=embed_dim, llm_dim=embed_dim)

        # 语言模型组件
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # 因果 Transformer
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

        # 权重共享
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
            image: (B, C, H, W) 输入图像
            text_ids: (B, T) 文本 token ids
            targets: (B, T) 标签（-100 表示不计算损失）
        """
        B = text_ids.shape[0]

        # 1. 视觉编码
        vision_feat = self.vision_encoder(image)  # (B, n_patches, embed_dim)
        vision_tokens = self.projector(vision_feat)  # (B, n_patches, embed_dim)

        # 2. 文本编码
        T = text_ids.shape[1]
        text_tokens = self.tok_emb(text_ids)  # (B, T, embed_dim)

        # 3. 拼接：[视觉token | 文本token]
        combined = torch.cat([vision_tokens, text_tokens], dim=1)  # (B, n_patches+T, embed_dim)

        # 4. 加位置编码
        seq_len = combined.shape[1]
        positions = torch.arange(seq_len, device=text_ids.device)
        combined = combined + self.pos_emb(positions)

        # 5. 因果 Transformer
        for block in self.blocks:
            combined = combined + block['attn'](block['norm1'](combined))
            combined = combined + block['ffn'](block['norm2'](combined))
        combined = self.norm(combined)

        # 6. 只对文本部分计算 logits（视觉 token 不需要预测词）
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
        """多模态生成：给定图像和文本 prompt，生成回答"""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(image, text_ids)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            text_ids = torch.cat([text_ids, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
            # 防止超出 max_len
            if text_ids.shape[1] >= self.max_len - self.n_patches:
                break
        return text_ids

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# 复用第21课的因果注意力和 FFN
class CausalSelfAttention(nn.Module):
    """单头因果自注意力"""

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
    """标准 FFN：Linear → GELU → Linear"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ============================================================================
# 5. 多模态数据集
# ============================================================================


class SimpleMultimodalDataset:
    """简化版多模态数据集

    每条数据包含：一张图像 + 一段文本对话
    图像用随机数据模拟，文本用字符级 tokenizer 编码
    """

    def __init__(self, data, img_size=32, max_length=96):
        self.data = data
        self.img_size = img_size
        self.max_length = max_length

        # 构建字符级词表
        all_chars = set()
        for sample in data:
            for key in ("prompt", "answer"):
                all_chars.update(sample[key])
        common_chars = " 。，！？；：、（）《》" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789"
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
        self.id_image = special_start + 5  # <image> 特殊 token

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

        self.vocab_size = len(chars) + 2 + 6  # +6 特殊 token

    def _encode_text(self, text):
        return [self.char2id.get(c, self.unk_token_id) for c in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 生成模拟图像（用固定种子保证同一样本每次图像一致）
        torch.manual_seed(idx + 1000)
        image = torch.randn(3, self.img_size, self.img_size)

        # 编码文本：[BOS_USER, prompt, EOS, BOS_AST, answer, EOS]
        prompt_ids = self._encode_text(sample["prompt"])
        answer_ids = self._encode_text(sample["answer"])

        text_ids = (
            self.bos_user_ids + prompt_ids + self.eos_ids +
            self.bos_assistant_ids + answer_ids + self.eos_ids
        )

        # 截断 / Padding
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
        else:
            text_ids = text_ids + [self.pad_token_id] * (self.max_length - len(text_ids))

        # 构造 labels：只对 assistant 部分计算损失
        labels = self._create_labels(text_ids)

        return image, torch.tensor(text_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _create_labels(self, text_ids):
        """和第21课一样的 Loss Mask 逻辑"""
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


# 创建模拟多模态数据
mm_data = [
    {"prompt": "描述这张图片。", "answer": "图片中有一只橘猫。"},
    {"prompt": "图里有什么颜色？", "answer": "有红色和蓝色。"},
    {"prompt": "这是什么场景？", "answer": "这是一个公园的场景。"},
    {"prompt": "图片中有几个人？", "answer": "图片中有两个人。"},
    {"prompt": "天气怎么样？", "answer": "天气晴朗，阳光明媚。"},
]

mm_dataset = SimpleMultimodalDataset(mm_data)
image, text_ids, labels = mm_dataset[0]
print(f"\n--- 多模态数据集示例 ---")
print(f"图像: {image.shape}  (C, H, W)")
print(f"文本: {text_ids.shape}  (max_length,)")
print(f"标签: {labels.shape}  (max_length,)")
print(f"需要计算损失的 token: {(labels != -100).sum().item()}")
print(f"词表大小: {mm_dataset.vocab_size}")

# ============================================================================
# 6. 训练多模态模型
# ============================================================================


def train_multimodal(model, dataset, epochs=5, lr=1e-3, batch_size=2, seed=42):
    """多模态训练循环"""
    if seed is not None:
        torch.manual_seed(seed)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 分组学习率：视觉编码器用更小的学习率（预训练权重更珍贵）
    vision_params = list(model.vision_encoder.parameters()) + list(model.projector.parameters())
    llm_params = list(model.tok_emb.parameters()) + list(model.lm_head.parameters())
    for block in model.blocks:
        llm_params += list(block.parameters())
    llm_params += list(model.pos_emb.parameters()) + list(model.norm.parameters())

    optimizer = torch.optim.AdamW([
        {"params": vision_params, "lr": lr * 0.1},  # 视觉部分学习率更小
        {"params": llm_params, "lr": lr},
    ], weight_decay=0.01)

    # 学习率调度
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


# 运行训练
print("\n--- 多模态模型训练 ---")
torch.manual_seed(42)
mm_model = SimpleMultimodalGPT(
    vocab_size=mm_dataset.vocab_size,
    img_size=32, patch_size=8,
    embed_dim=64, n_layers=2, max_len=128
)
print(f"模型参数量: {mm_model.num_params() / 1e3:.1f}K")

print("\n[训练开始]")
mm_model = train_multimodal(mm_model, mm_dataset, epochs=8, lr=1e-3, batch_size=2, seed=42)

# ============================================================================
# 7. 推理演示
# ============================================================================
print("\n--- 多模态推理演示 ---")
mm_model.eval()

# 模拟一张新图像
test_image = torch.randn(1, 3, 32, 32)
test_prompt = "描述这张图片。"
prompt_ids = mm_dataset._encode_text(test_prompt)
start_ids = mm_dataset.bos_user_ids + prompt_ids + mm_dataset.eos_ids + mm_dataset.bos_assistant_ids
text_input = torch.tensor([start_ids], dtype=torch.long)

generated = mm_model.generate(test_image, text_input, max_new_tokens=15, temperature=0.8, eos_id=mm_dataset.id_eos)
decoded = "".join(mm_dataset.id2char.get(int(t), '?') for t in generated[0])
print(f"Prompt: {test_prompt}")
print(f"生成结果: {decoded}")

# ============================================================================
# 8. 多模态训练的关键技巧
# ============================================================================
"""
1. 视觉编码器通常冻结 (freeze)
   - 预训练的 ViT 已经很强，不需要再训练
   - 只训练 Projector + LLM
   - 节省显存和训练时间

2. 分阶段训练
   - 阶段1: 冻结 ViT + LLM，只训练 Projector（对齐视觉和语言空间）
   - 阶段2: 解冻 LLM，联合微调（提升多模态理解能力）
   - 阶段3: 全部解冻，小学习率精调（可选）

3. 分组学习率
   - ViT: 最小学习率（1e-6），因为预训练权重最珍贵
   - Projector: 中等学习率（1e-4），需要快速对齐
   - LLM: 较大学习率（1e-5），需要适应多模态输入

4. 图像分辨率
   - 训练时用低分辨率 (224×224) 节省显存
   - 推理时可用高分辨率提升效果
   - 任何分辨率都需要是 patch_size 的整数倍

5. 数据质量
   - 图文对齐是关键：图像和描述必须匹配
   - 数据量不需要太大，但质量要高
   - 常用数据集: COCO Captions, LAION, ShareGPT4V
"""

# ============================================================================
# 9. 冻结视觉编码器实验
# ============================================================================
print("\n--- 冻结视觉编码器实验 ---")

torch.manual_seed(42)
frozen_model = SimpleMultimodalGPT(
    vocab_size=mm_dataset.vocab_size,
    img_size=32, patch_size=8,
    embed_dim=64, n_layers=2, max_len=128
)

# 冻结视觉编码器
for param in frozen_model.vision_encoder.parameters():
    param.requires_grad = False

# 统计可训练参数
total_params = frozen_model.num_params()
trainable_params = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
print(f"总参数: {total_params / 1e3:.1f}K")
print(f"可训练参数: {trainable_params / 1e3:.1f}K ({trainable_params / total_params * 100:.0f}%)")
print(f"冻结参数: {(total_params - trainable_params) / 1e3:.1f}K ({(1 - trainable_params / total_params) * 100:.0f}%)")

print("\n[冻结 ViT 训练]")
frozen_model = train_multimodal(frozen_model, mm_dataset, epochs=5, lr=1e-3, batch_size=2, seed=42)

# ============================================================================
# 10. 多模态 vs 纯文本对比
# ============================================================================
print("\n--- 多模态 vs 纯文本模型对比 ---")

# 纯文本模型（没有视觉输入）
class TextOnlyGPT(nn.Module):
    """纯文本 GPT（用于对比）"""

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

print(f"多模态模型参数: {mm_model.num_params() / 1e3:.1f}K")
print(f"纯文本模型参数: {text_model.num_params() / 1e3:.1f}K")
print(f"视觉编码器占比: {sum(p.numel() for p in mm_model.vision_encoder.parameters()) / mm_model.num_params() * 100:.0f}%")

# ============================================================================
# 练习
# ============================================================================
print("\n" + "=" * 70)
print("练习（请先自己思考/动手，再看下面的参考答案）")
print("=" * 70)

# 练习1：修改融合方式
print("\n练习1：修改融合方式")
print("  当前：[视觉token] + [文本token] 拼接融合")
print("  请改为：[文本token] + [视觉token] + [文本token] 三段拼接")
print("  即：prompt文本 → 图像 → 回答文本")
print("  提示：修改 SimpleMultimodalGPT.forward 中的 torch.cat 顺序")
print("  思考：这种顺序有什么好处？和 LLaVA 的做法有什么区别？")

# 练习2：冻结不同组件
print("\n练习2：冻结不同组件")
print("  当前：冻结了视觉编码器")
print("  请尝试：")
print("  (a) 冻结 LLM，只训练 Projector + ViT")
print("  (b) 全部冻结，只训练 Projector")
print("  (c) 全部不冻结")
print("  对比四种策略的训练 loss 和生成质量。")

# 练习3：添加更多图像增强
print("\n练习3：图像增强")
print("  当前：用随机噪声模拟图像")
print("  请在 __getitem__ 中添加简单的图像增强：")
print("  - 随机水平翻转")
print("  - 随机裁剪")
print("  - 颜色抖动")
print("  提示：用 torchvision.transforms 或手动实现")

# 练习4：思考题
print("\n练习4：为什么视觉编码器用双向注意力，而语言模型用因果注意力？")
print("  ViT 的 TransformerEncoderLayer 是双向的（能看到所有 patch），")
print("  而 GPT 的 CausalSelfAttention 是因果的（只能看到前面的 token）。")
print("  为什么这样设计？如果 ViT 也用因果注意力会怎样？")

# 练习4 参考答案
print("\n[参考答案 · 练习4]")
print("  图像是完整的——所有 patch 同时存在，没有时序关系，")
print("  所以 ViT 用双向注意力让每个 patch 看到整张图是合理的。")
print("  而文本是逐 token 生成的——当前 token 不能『偷看』未来的 token，")
print("  所以 LLM 必须用因果注意力。")
print("  如果 ViT 用因果注意力，每个 patch 只能看到左上方的 patch，")
print("  会严重损害视觉理解能力（右下角的物体就『看不见』左上角的上下文了）。")

print("\n" + "=" * 70)
print("第26课总结：")
print("  1. 多模态模型 = 视觉编码器 + 投影器 + 语言模型")
print("  2. ViT 把图像切成 patch，编码为向量序列")
print("  3. 投影器把视觉特征映射到语言模型的嵌入空间")
print("  4. 融合方式：拼接（最简单）、交叉注意力（更灵活）")
print("  5. 训练技巧：冻结 ViT、分阶段训练、分组学习率")
print("  6. 视觉编码器用双向注意力，语言模型用因果注意力")
print("=" * 70)
