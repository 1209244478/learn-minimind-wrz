# ============================================================================
# 第21课：监督微调 (Supervised Fine-Tuning, SFT)
# ============================================================================
"""
上一课我们完成了从零搭建迷你 LLM 的综合实战。但预训练模型只会"续写"，
不会"对话"。SFT 的目标就是教模型学会按照指令回答问题。

本课内容：
1. 为什么需要 SFT？预训练 vs 微调的区别
2. SFT 数据格式：对话模板 (Chat Template)
3. SFT 的标签构造：只对回答部分计算损失
4. SFT 训练循环实现
5. 学习率与训练策略：小学习率 + 短训练

关键概念：
- 预训练 (Pretrain)：让模型学会语言的统计规律（续写能力）
- 监督微调 (SFT)：让模型学会按指令回答（对话能力）
- Chat Template：将多轮对话格式化为模型可理解的文本
- Loss Mask：只对 assistant 回答部分计算损失，忽略 prompt 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# 全局随机种子（保证本课所有实验可复现）
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# 1. 为什么需要 SFT？
# ============================================================================
"""
预训练模型 = 续写机器：
  输入: "中国的首都是"
  输出: "北京，位于华北平原..."  ← 续写，不是回答

SFT 模型 = 对话助手：
  输入: "<|im_start|>user\n中国的首都是哪里？<|im_end|>\n<|im_start|>assistant\n"
  输出: "中国的首都是北京。"  ← 按指令回答

关键区别：
┌──────────┬──────────────────┬──────────────────┐
│          │ 预训练 (Pretrain) │ 微调 (SFT)       │
├──────────┼──────────────────┼──────────────────┤
│ 目标     │ 学会语言规律      │ 学会按指令回答    │
│ 数据     │ 纯文本            │ 指令-回答对       │
│ 损失     │ 全部 token        │ 只对回答部分      │
│ 学习率   │ 较大 (1e-4)       │ 很小 (1e-5)      │
│ 训练轮数 │ 多 (3-10 epochs)  │ 少 (1-3 epochs)  │
└──────────┴──────────────────┴──────────────────┘
"""

print("=" * 70)
print("第21课：监督微调 (SFT)")
print("=" * 70)

# ============================================================================
# 2. Chat Template — 对话模板
# ============================================================================
"""
不同模型使用不同的对话模板，将多轮对话转为模型可读的文本。

ChatML 格式（Qwen/MiniMind 使用）：
<|im_start|>system
你是一个有用的AI助手。<|im_end|>
<|im_start|>user
你好，请介绍一下自己。<|im_end|>
<|im_start|>assistant
你好！我是MiniMind...<|im_end|>

Llama 格式：
[INST] <<SYS>>
你是一个有用的AI助手。
<</SYS>>
你好，请介绍一下自己。 [/INST] 你好！我是Llama...

关键点：
- 特殊 token（<|im_start|>, <|im_end|>）标记角色边界
- 模型只在 assistant 部分生成内容
- system prompt 设定模型行为
"""

def apply_chat_template(messages):
    """简单的 ChatML 模板实现

    Args:
        messages: 对话列表，如 [{"role": "user", "content": "你好"}, ...]

    Returns:
        格式化后的对话文本
    """
    # ChatML 格式（Qwen/MiniMind 使用）：<|im_start|>role\ncontent<|im_end|>\n
    BOS = "<|im_start|>"
    EOS = "<|im_end|>\n"

    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{BOS}{role}\n{content}{EOS}")

    return "".join(parts)


# 演示 Chat Template
messages = [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "中国的首都是哪里？"},
    {"role": "assistant", "content": "中国的首都是北京。"},
]

formatted = apply_chat_template(messages)
print("\n--- Chat Template 示例 ---")
print(formatted)

# ============================================================================
# 3. SFT 标签构造 — Loss Mask
# ============================================================================
"""
SFT 的核心：只对 assistant 的回答部分计算损失！

为什么？
- prompt 部分（system + user）是输入，模型不需要学习生成它们
- 只有 assistant 的回答才是模型需要学习的内容

实现方式：
- 将整个对话 tokenize 为 input_ids
- labels 中，非 assistant 部分设为 -100（ignore_index）
- assistant 回答部分 labels = input_ids（正常计算损失）

示例：
  input_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  labels:    [-100,-100,-100,-100, 5, 6, 7, 8, 9, 10]
              ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
              system+user (不计算)   assistant (计算)
"""


def create_sft_labels(input_ids, bos_assistant_ids, eos_ids, pad_token_id=0):
    """为 SFT 数据创建标签，只对 assistant 部分计算损失

    采用单次扫描 + 状态机实现，复杂度 O(N)。
    一旦进入 assistant 区间，就一直标记直到遇到 EOS；
    padding 位置的 label 保持 -100（不参与损失）。

    Args:
        input_ids: 完整对话的 token id 列表
        bos_assistant_ids: "<|im_start|>assistant\n" 的 token ids
        eos_ids: "<|im_end|>" 的 token ids
        pad_token_id: padding token 的 id

    Returns:
        labels: 与 input_ids 等长的标签列表，非 assistant 部分为 -100
    """
    n = len(input_ids)
    labels = [-100] * n
    bsz = len(bos_assistant_ids)
    esz = len(eos_ids)
    i = 0

    while i < n:
        # 命中 assistant 起始标记：进入 assistant 区间
        if i + bsz <= n and input_ids[i:i + bsz] == bos_assistant_ids:
            start = i + bsz
            j = start
            # 在区间内寻找 EOS
            while j < n and not (j + esz <= n and input_ids[j:j + esz] == eos_ids):
                j += 1
            # 标记 [start, j + esz) 区间为有效（EOS 自身也要学习生成）
            for k in range(start, min(j + esz, n)):
                if input_ids[k] != pad_token_id:
                    labels[k] = input_ids[k]
            i = j + esz
        else:
            i += 1

    return labels


# 演示标签构造
print("\n--- SFT 标签构造示例 ---")
# 模拟一个简单的 token 序列
# 假设: [BOS_SYS] system内容 [EOS] [BOS_USR] user内容 [EOS] [BOS_AST] 回答 [EOS] [PAD]...
simple_input = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 0, 0]
# BOS_AST = [30], EOS = [34]
labels = create_sft_labels(simple_input, bos_assistant_ids=[30], eos_ids=[34], pad_token_id=0)

print(f"input_ids: {simple_input}")
print(f"labels:    {labels}")
print(f"计算损失的位置: {[i for i, l in enumerate(labels) if l != -100]}")

# ============================================================================
# 4. SFT 数据集
# ============================================================================


class SimpleSFTDataset:
    """SFT 数据集

    使用字符级 Tokenizer 对对话文本进行编码，
    构造 input_ids 和 labels（仅 assistant 部分计算损失）。

    关键修正：用不同的 BOS 区分 user 段和 assistant 段，
    这样 create_sft_labels 只会匹配到 assistant 段，
    不会把 user 的 prompt 错误地纳入损失计算。
    """

    def __init__(self, data, max_length=64, shared_vocab=None):
        self.data = data
        self.max_length = max_length

        if shared_vocab is not None:
            # 复用已有词表（验证集必须与训练集共享同一词表）
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

        # 构建字符级词表：训练数据 + ChatML 特殊字符 + 常用字符
        all_chars = set()
        for sample in data:
            for key in ("content", "answer"):
                all_chars.update(sample[key])
        # 补全 ChatML 特殊字符与常见中英文字符，避免 UNK
        # （演示数据集很小，加常用字符集可保证新输入也能被正确编码）
        common_chars = " 。，！？；：、（）《》" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789"
        special_chars = "<|im_start|>system\nuser\nassistant\n<|im_end|>"
        all_chars.update(special_chars)
        all_chars.update(common_chars)
        chars = sorted(all_chars)

        self.pad_token_id = 0
        self.unk_token_id = 1
        # 特殊 token id 分配（在普通字符之后连续分配）：
        #   0      -> <pad>
        #   1      -> <unk>
        #   [2, 2+len(chars)) -> 字符
        #   +0     -> <|im_start|>       （共享 token）
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

        # 关键：用 id_user 拼 user 段，用 id_assistant 拼 assistant 段
        # 这样 create_sft_labels 只匹配 assistant 段
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

        # 词表大小 = pad + unk + 普通字符 + 5 个特殊 token
        self.vocab_size = len(chars) + 2 + 5

    def get_vocab(self):
        """导出词表信息，供验证集/测试集复用"""
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
        """用字符级 tokenizer 编码文本"""
        return [self.char2id.get(c, self.unk_token_id) for c in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """返回 (input_ids, labels) 对

        序列结构：
          [BOS_USER, prompt_tokens, EOS, BOS_AST, answer_tokens, EOS, PAD...]
        """
        sample = self.data[idx]

        prompt_ids = self._encode_text(sample["content"])
        answer_ids = self._encode_text(sample["answer"])

        # 拼接：user 段 + assistant 段（用不同的 BOS 标记）
        input_ids = (
            self.bos_user_ids + prompt_ids + self.eos_ids +
            self.bos_assistant_ids + answer_ids + self.eos_ids
        )

        # 截断或 Padding
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids = input_ids + [self.pad_token_id] * (self.max_length - len(input_ids))

        # 构造 labels：只对 assistant 回答部分计算损失
        labels = create_sft_labels(input_ids, self.bos_assistant_ids, self.eos_ids, self.pad_token_id)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# 创建模拟 SFT 数据
sft_data = [
    {"role": "user", "content": "什么是机器学习？", "answer": "机器学习是AI的一个分支..."},
    {"role": "user", "content": "Python是什么？", "answer": "Python是一种编程语言..."},
    {"role": "user", "content": "深度学习和机器学习的区别？", "answer": "深度学习是机器学习的子集..."},
    {"role": "user", "content": "你好，请介绍一下自己。", "answer": "你好！我是 MiniMind，一个迷你语言模型。"},
    {"role": "user", "content": "1+1等于几？", "answer": "1+1=2。"},
]

# 留出 1 条做验证集，剩下 4 条做训练集
sft_train_data = sft_data[:4]
sft_val_data = sft_data[4:]

sft_dataset = SimpleSFTDataset(sft_train_data)
sft_val_dataset = SimpleSFTDataset(sft_val_data, max_length=sft_dataset.max_length,
                                   shared_vocab=sft_dataset.get_vocab()) if sft_val_data else None

input_ids, labels = sft_dataset[0]
print(f"\n--- SFT 数据集示例 ---")
print(f"input_ids shape: {input_ids.shape}")
print(f"labels shape: {labels.shape}")
print(f"需要计算损失的 token 数: {(labels != -100).sum().item()}")
print(f"总 token 数: {labels.shape[0]}")
# 把 input_ids 反解回字符，直观检查序列结构
print(f"解码前 20 个 token: {''.join(sft_dataset.id2char.get(int(t), '?') for t in input_ids[:20])}")

# ============================================================================
# 5. SFT 训练循环
# ============================================================================


class CausalSelfAttention(nn.Module):
    """单头因果自注意力（教学简化版）

    关键点：用上三角 mask 保证 token i 只能看到 [0, i] 的 token，
    这是 GPT 类模型与双向 Transformer 的本质区别。
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
        # 注意力分数 + 缩放
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, T, T)
        # 因果 mask：上三角（含对角线以上）置为 -inf
        # 注：教学版每次重建 mask；生产代码应预计算并缓存
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # (B, T, head_dim)
        return self.o_proj(out)


class SimpleFFN(nn.Module):
    """标准 FFN：Linear -> GELU -> Linear（教学版，使用 GELU 保持简洁）"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SimpleBlock(nn.Module):
    """Pre-norm Transformer Block：Attention + FFN"""

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
    """简化 GPT（演示用）

    与原实现的区别（修复 #2）：
      - 不再使用 nn.TransformerEncoderLayer（双向注意力）
      - 采用手写的因果自注意力 + Pre-norm + 权重共享
      - 真正可作为 next-token-prediction 的 LM 使用
    """

    def __init__(self, vocab_size, dim=64, n_layers=2, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        # 单头注意力（head_dim == dim），减少参数，方便演示
        self.head_dim = dim

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([SimpleBlock(dim, self.head_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # 权重共享：Embedding 与 LM Head 共用参数
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
            # 关键：ignore_index=-100 让非 assistant 部分不参与损失计算
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=20, temperature=1.0, eos_id=None):
        """朴素自回归生成（贪婪截断到 max_len）"""
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
    """SFT 训练循环

    与预训练的关键区别：
    1. 学习率更小（1e-5 vs 1e-4）
    2. 训练轮数更少（1-3 vs 3-10）
    3. 损失只计算 assistant 部分
    4. 通常从预训练权重开始（而非随机初始化）

    Args:
        model: 待训练模型
        dataset: 训练集
        epochs: 训练轮数
        lr: 学习率（SFT 常用 1e-5 ~ 5e-5）
        batch_size: batch 大小
        val_dataset: 验证集（可选，提供后每 log_interval 个 epoch 计算一次 val loss）
        warmup_ratio: 学习率线性预热步数占总步数的比例
        max_grad_norm: 梯度裁剪阈值
        log_interval: 每 N 个 epoch 打印一次日志
        seed: 随机种子（用于 DataLoader 与模型初始化的可复现性）
        verbose: 是否打印日志
    """
    if seed is not None:
        torch.manual_seed(seed)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=len(dataset) >= batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 学习率调度器：线性预热 + 余弦衰减（标准 SFT 配置）
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


# 运行 SFT 训练
print("\n--- SFT 训练演示 ---")
# 修复 #4：固定随机种子保证可复现
torch.manual_seed(42)
model = SimpleGPT(vocab_size=sft_dataset.vocab_size, dim=64, n_layers=2)
print(f"模型参数量: {model.num_params() / 1e3:.1f}K")
print(f"词表大小: {sft_dataset.vocab_size}")

# 模拟：先用"预训练"初始化，再做 SFT
print("\n[SFT 训练开始]")
model = train_sft(model, sft_dataset, epochs=8, lr=1e-3, batch_size=2,
                  val_dataset=sft_val_dataset, warmup_ratio=0.1, seed=42)

# ============================================================================
# 6. SFT vs 预训练的损失对比
# ============================================================================
"""
让我们对比一下 SFT 和预训练在损失计算上的区别：
"""

print("\n--- SFT vs 预训练 损失对比 ---")

# 模拟一个 batch 的数据
torch.manual_seed(42)
B, T, V = 2, 16, 100
logits = torch.randn(B, T, V)

# 方式1：预训练 — 全部 token 都计算损失
pretrain_labels = torch.randint(0, V, (B, T))
pretrain_loss = F.cross_entropy(logits.view(-1, V), pretrain_labels.view(-1))
print(f"预训练损失（全部 token）: {pretrain_loss.item():.4f}")

# 方式2：SFT — 只对 assistant 部分计算损失
sft_labels = pretrain_labels.clone()
# 假设前 10 个 token 是 prompt，设为 -100
sft_labels[:, :10] = -100
sft_loss = F.cross_entropy(logits.view(-1, V), sft_labels.view(-1), ignore_index=-100)
n_valid = (sft_labels != -100).sum().item()
n_total = sft_labels.numel()
print(f"SFT 损失（仅 assistant）: {sft_loss.item():.4f}")
print(f"参与计算的 token: {n_valid}/{n_total} ({n_valid / n_total * 100:.0f}%)")

# ============================================================================
# 6.5  Loss Mask 效果对比：训练 vs 生成
# ============================================================================
"""
修复 #8：把"有/无 Loss Mask"的训练效果用生成结果直观呈现。

方法：训练两个对照模型，让它们从同一个 prompt 续写。
  - 模型 A：用 SFT 标签训练（只对 assistant 部分算 loss）— 正确
  - 模型 B：用全量标签训练（对包括 user prompt 在内的所有 token 算 loss）— 错误
预期：模型 A 学会"按指令回答"；模型 B 学会"复述问题后乱答"。
"""


class _AllLabelsDataset(SimpleSFTDataset):
    """对照组数据集：labels == input_ids，对每个 token 都算 loss"""
    def __getitem__(self, idx):
        ids, _ = super().__getitem__(idx)
        return ids, ids.clone()


print("\n--- Loss Mask 效果对比（生成式）---")
unmasked_dataset = _AllLabelsDataset(sft_train_data, max_length=sft_dataset.max_length)
unmasked_val = _AllLabelsDataset(sft_val_data, max_length=sft_dataset.max_length,
                                 shared_vocab=sft_dataset.get_vocab()) if sft_val_data else None

torch.manual_seed(42)
model_unmasked = SimpleGPT(vocab_size=sft_dataset.vocab_size, dim=64, n_layers=2)
print("[对照组训练：所有 token 都算 loss]")
model_unmasked = train_sft(model_unmasked, unmasked_dataset, epochs=8, lr=1e-3,
                           batch_size=2, val_dataset=unmasked_val, warmup_ratio=0.1,
                           seed=42, verbose=True)

test_prompt = "什么是机器学习？"
prompt_ids = sft_dataset._encode_text(test_prompt)
# 直接用 [BOS_USER, prompt..., EOS, BOS_AST] 作为起点，让模型接着生成
start_ids = (sft_dataset.bos_user_ids + prompt_ids + sft_dataset.eos_ids + sft_dataset.bos_assistant_ids)
start_tensor = torch.tensor([start_ids], dtype=torch.long)


def gen_text(model, ids, max_new=20):
    out = model.generate(ids, max_new_tokens=max_new, temperature=0.8,
                         eos_id=sft_dataset.id_eos)
    return "".join(sft_dataset.id2char.get(int(t), '?') for t in out[0])


print(f"\nPrompt: {test_prompt}")
print(f"  正确（SFT Mask）：  {gen_text(model, start_tensor.clone())}")
print(f"  错误（无 Mask）：   {gen_text(model_unmasked, start_tensor.clone())}")

# ============================================================================
# 7. SFT 的关键训练技巧
# ============================================================================
"""
1. 学习率要小（1e-5 ~ 5e-5）
   - 预训练权重已经很珍贵，大学习率会"遗忘"
   - 这就是"灾难性遗忘" (Catastrophic Forgetting)

2. 训练轮数要少（1-3 epochs）
   - SFT 数据量远小于预训练数据
   - 训练太多轮会导致过拟合

3. 使用余弦退火学习率
   - 从 lr 预热到峰值，再缓慢下降

4. 梯度裁剪
   - 防止梯度爆炸，通常 clip=1.0

5. 权重衰减 (Weight Decay)
   - AdamW 的 weight_decay=0.01，防止过拟合

6. 从预训练权重开始
   - 绝不要从随机初始化开始 SFT！
   - 应该加载预训练好的权重，然后小学习率微调
"""

# ============================================================================
# 8. 完整的 SFT Pipeline
# ============================================================================
"""
完整的 SFT 流程：

1. 数据准备
   - 收集指令-回答对数据
   - 格式化为 ChatML 对话格式
   - 划分训练/验证集

2. 数据预处理
   - 用 tokenizer 编码对话
   - 构造 labels（只对 assistant 部分计算损失）
   - Padding 和截断

3. 训练
   - 加载预训练权重
   - 小学习率 (1e-5) + AdamW
   - 余弦退火学习率调度
   - 梯度裁剪 (1.0)
   - 混合精度训练 (bf16/fp16)

4. 评估
   - 在验证集上计算损失
   - 人工评估生成质量
   - 使用 GPT-4 等模型自动评估

5. 保存
   - 保存 SFT 后的权重
   - 可用于后续 DPO/RLHF 训练
"""

# ============================================================================
# 练习
# ============================================================================
print("\n" + "=" * 70)
print("练习（请先自己思考/动手，再看下面的参考答案）")
print("=" * 70)

# 练习1（思考题，不给完整答案）：构造 SFT 标签
print("\n练习1：构造 SFT 标签")
print("给定以下 token 序列：")
test_input = [1, 5, 6, 7, 50, 51, 30, 31, 32, 2, 0, 0]
print(f"  input_ids: {test_input}")
print(f"  BOS_AST = [50, 51], EOS = [2], PAD = 0")
print("请手写出 labels（提示：assistant 区间有几个 token？padding 怎么处理？）")
# 留一个验证函数，不直接 print 答案
def _check_ex1(student_labels):
    expected = create_sft_labels(test_input, [50, 51], [2], 0)
    return student_labels == expected, expected
print("  验证方式：把答案赋给 my_labels，调用 _check_ex1(my_labels)")
print("  示例：ok, exp = _check_ex1([-100]*12)  # 把答案填进列表里")

# 练习2（动手题）：改变学习率重新训练，观察 Loss 曲线
print("\n练习2：学习率对 SFT 的影响")
print("请修改下面的 lr / epochs，重新跑训练并比较 Loss 曲线：")
print("  提示：把 lr 设为 1e-1、1e-3、1e-5 各跑一次，看看哪个收敛、哪个发散。")
print("  代码模板：")
print("    for lr_ in [1e-1, 1e-3, 1e-5]:")
print("        torch.manual_seed(42)")
print("        m = SimpleGPT(vocab_size=sft_dataset.vocab_size, dim=64, n_layers=2)")
print("        train_sft(m, sft_dataset, epochs=5, lr=lr_, batch_size=2, seed=42)")
print("  思考：哪种学习率会触发灾难性遗忘？哪种训练不动？")

# 练习3（动手题）：构造多轮对话的 ChatML
print("\n练习3：把多轮对话喂给 SFT 数据集")
print("  现状：sft_data 每条只有一轮 user+assistant。")
print("  请改写 __getitem__，让 input_ids 包含 2 轮 user/assistant（多轮对话），")
print("  并保证 create_sft_labels 只对两段 assistant 都算 loss。")
print("  提示：把 bos_user_ids / eos_ids / bos_assistant_ids 交替拼起来即可。")

# 练习4（思考题）：为什么用因果 mask
print("\n练习4：为什么 SimpleGPT 一定要用因果 mask？")
print("  如果把 CausalSelfAttention.forward 里的 mask 去掉（变成全注意力），")
print("  SFT 训练会发生什么？训练 loss 和生成结果会怎样？")
print("  提示：想想 token 0 已经『看到』了 token 5 的答案，")
print("       训练时 label 又告诉它『在 token 0 之后应该是 X』，会学成什么？")

# 示例代码：练习4的答案实验
print("\n[参考答案 · 练习4 实验]")
class BiasedAttention(CausalSelfAttention):
    """去掉 mask 的双向注意力（用于对比实验）"""
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # 故意不加 causal mask —— 这就是『错』的版本
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
print("  双向注意力 SFT 训练完成（注意 val loss 与正常模型的差异）")

# 练习4答案的多轮 ChatML 示例
print("\n[参考答案 · 练习3 示例]")
multi_turn = [
    {"role": "system", "content": "你是数学老师。"},
    {"role": "user", "content": "1+1等于几？"},
    {"role": "assistant", "content": "1+1=2。"},
    {"role": "user", "content": "那2+2呢？"},
    {"role": "assistant", "content": "2+2=4。"},
]
result = apply_chat_template(multi_turn)
print(f"  格式化结果（多轮 ChatML）：\n  {result.replace(chr(10), chr(10) + '  ')}")

print("\n" + "=" * 70)
print("第21课总结：")
print("  1. SFT 的目标：让预训练模型学会按指令回答")
print("  2. Chat Template：将多轮对话格式化为模型可读文本")
print("  3. Loss Mask：只对 assistant 回答部分计算损失（ignore_index=-100）")
print("  4. 关键技巧：小学习率 + 少轮数 + 从预训练权重开始")
print("  5. 灾难性遗忘：学习率过大会破坏预训练知识")
print("=" * 70)
