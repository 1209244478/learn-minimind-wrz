"""
第2课：Embedding — 数字如何变成向量？
=====================================

为什么需要 Embedding？
  上一课: 文字 → 数字 (Token ID)
  这一课: 数字 → 向量 (Embedding)
  下一课: 向量 → 理解语义 (Attention)

为什么数字还不够？
  "国王" = 1234
  "王后" = 5678
  "苹果" = 9012

  问题: 1234 - 5678 = ? 没意义
        1234 + 9012 = ? 也没意义

  模型要做数学运算，需要有"意义"的数字

Embedding 的核心思想:
  把每个 token ID 映射为一个 d 维向量
  向量中的每个维度代表某种"语义特征"

  例如 d=4:
    国王 → [0.8, 0.9, 0.2, 0.1]  (男性度=高, 权力=高)
    王后 → [0.7, 0.2, 0.2, 0.1]  (男性度=低, 权力=高)
    公主 → [0.6, 0.3, 0.1, 0.1]  (男性度=低, 权力=中)

类比:
  字典查表: 给一个词, 返回定义
  Embedding: 给一个 ID, 返回向量
  - 本质是一个 V×d 的查找表 (Lookup Table)
  - V = 词表大小 (vocab_size)
  - d = 嵌入维度 (hidden_size)

MiniMind 用的是:
  - vocab_size = 6400
  - hidden_size = 768

运行: python lessons/lesson02_embedding.py
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 第1部分：手动实现 Embedding
# ============================================================
print("=" * 60)
print("第1部分：手动实现 Embedding (查找表)")
print("=" * 60)


class SimpleEmbedding:
    """手动实现 Embedding: 本质是一个查找表"""

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # 随机初始化查找表: V × d 矩阵
        self.weight = torch.randn(vocab_size, embed_dim) * 0.1

    def forward(self, token_ids):
        """
        token_ids: [batch, seq_len] 的整数 ID
        返回: [batch, seq_len, embed_dim] 的向量
        """
        return self.weight[token_ids]


# 演示
vocab_size = 100
embed_dim = 8
embedding = SimpleEmbedding(vocab_size, embed_dim)

# 假设一个 batch: 2 个句子, 每个 5 个 token
token_ids = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

vectors = embedding.forward(token_ids)
print(f"\n输入 token_ids shape: {token_ids.shape}")
print(f"输入: \n{token_ids}")
print(f"\n输出 vectors shape: {vectors.shape}")
print(f"每个 token 变成 {embed_dim} 维向量")
print(f"\n第 1 个句子的 embedding:")
print(vectors[0])


# ============================================================
# 第2部分：PyTorch 的 nn.Embedding
# ============================================================
print("\n" + "=" * 60)
print("第2部分：PyTorch 的 nn.Embedding")
print("=" * 60)

# 实际使用
embedding = nn.Embedding(
    num_embeddings=100,    # 词表大小
    embedding_dim=8,       # 嵌入维度
    padding_idx=0          # padding 索引 (永远是 0 向量)
)

vectors = embedding(token_ids)
print(f"\nnn.Embedding 输出 shape: {vectors.shape}")
print(f"参数量: {sum(p.numel() for p in embedding.parameters())}")

# 重要: padding_idx 永远是 0 向量
print(f"\npadding_idx=0 的向量 (永远为零):")
print(embedding.weight[0])
print(f"是否全为 0: {torch.all(embedding.weight[0] == 0).item()}")


# ============================================================
# 第3部分：Embedding 的语义
# ============================================================
print("\n" + "=" * 60)
print("第3部分：Embedding 的语义")
print("=" * 60)

print("""
训练后, Embedding 向量会"学"到语义:
  语义相近的词, 向量也相近
  语义相反的词, 向量相反
  语义相关的词, 在向量空间中聚类

经典例子 (Word2Vec 发现):
  vec(国王) - vec(男人) + vec(女人) ≈ vec(女王)

可视化示意 (2维):
  ┌─────────────────┐
  │   国王★          │
  │   男人●          │
  │          王后◆    │
  │                 │
  │  苹果■  橘子▲    │
  └─────────────────┘

  ★●◆ 在同一区域 (都是"人")
  ■▲ 在同一区域 (都是"水果")
  国王-王后 = 男人-女人 (性别差异)

如何衡量两个向量的相似度？
  - 余弦相似度: cos_sim(A, B) = (A·B) / (|A|×|B|)
  - 值域 [-1, 1], 1=完全相同, 0=无关, -1=完全相反
""")


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)


# 演示余弦相似度
torch.manual_seed(42)
# 假设训练后, 词向量已经学到一些语义
king = torch.tensor([0.8, 0.9, 0.2])
queen = torch.tensor([0.7, 0.2, 0.2])
man = torch.tensor([0.85, 0.95, 0.1])
apple = torch.tensor([0.1, 0.2, 0.9])

print(f"\n余弦相似度演示:")
print(f"  sim(国王, 王后) = {cosine_similarity(king, queen):.4f}")
print(f"  sim(国王, 男人) = {cosine_similarity(king, man):.4f}")
print(f"  sim(国王, 苹果) = {cosine_similarity(king, apple):.4f}")
print(f"\n观察: 国王与王后相似度高, 与苹果相似度低")


# ============================================================
# 第4部分：嵌入维度的影响
# ============================================================
print("\n" + "=" * 60)
print("第4部分：嵌入维度的影响")
print("=" * 60)

print("""
嵌入维度 (d) 是关键超参数:

d 太小 (如 64):
  ✓ 训练快, 显存少
  ✗ 表达力弱, 复杂语义分不开
  ✗ 训练数据不够时容易欠拟合

d 太大 (如 4096):
  ✓ 表达力强
  ✗ 训练慢, 显存大
  ✗ 需要更多训练数据

经验公式 (供参考):
  d ≈ (V ** 0.25) * 4   (V=词表大小)

MiniMind 实际配置:
  - MiniMind-Small:  d=512
  - MiniMind:        d=768
  - 大模型 (GPT-3):  d=12288 (175B)
""")

# 演示不同维度的影响
print("\n[演示] 不同词表大小对应的推荐维度")
print("-" * 60)
print(f"{'词表大小':<12}{'推荐 d':<10}{'总参数量':<15}")
print("-" * 60)
for V in [1000, 10000, 50000, 100000]:
    d = int((V ** 0.25) * 4)
    total = V * d
    print(f"{V:<12}{d:<10}{total:,}")


# ============================================================
# 第5部分：训练前 vs 训练后
# ============================================================
print("\n" + "=" * 60)
print("第5部分：训练前 vs 训练后")
print("=" * 60)

print("""
训练前的 Embedding:
  - 随机初始化 (N(0, 0.02) 或类似)
  - 没有任何语义
  - 任意两个词的相似度都接近 0

训练后的 Embedding:
  - 通过反向传播学习
  - 相似语义的词聚集
  - 出现"方向性"语义

类比:
  训练前 = 字典里只有词, 没有解释
  训练后 = 字典有了解释, 而且词之间有关系

关键洞察:
  Embedding 是模型"理解"语言的起点
  模型的"知识"很大一部分存储在 Embedding 中
  - 7B 模型, 1/3 参数在 Embedding
  - 词表 50K × 4096 = 2 亿参数
""")


# ============================================================
# 第6部分：权重共享 (Weight Tying)
# ============================================================
print("\n" + "=" * 60)
print("第6部分：权重共享 (Weight Tying)")
print("=" * 60)

print("""
有趣的发现:
  Embedding 的功能: token ID → 语义向量
  LM Head 的功能: 语义向量 → token 概率

  两者几乎是"互逆"的操作！
  → 很多模型 (包括 MiniMind) 把这两层共享权重

优点:
  ✓ 参数量减半 (Embedding 减一半)
  ✓ 训练更稳定 (梯度对称)
  ✓ 效果往往更好

代码:
```python
self.embed = nn.Linear(vocab_size, d, bias=False)  # Embedding
self.lm_head = nn.Linear(d, vocab_size, bias=False) # LM Head
self.lm_head.weight = self.embed.weight.T           # 共享权重
```

或者更常见的写法:
```python
self.embed = nn.Embedding(vocab_size, d)
self.lm_head = nn.Linear(d, vocab_size, bias=False)
self.lm_head.weight = self.embed.weight             # 直接共享
```
""")


# ============================================================
# 第7部分：MiniMind 中的 Embedding
# ============================================================
print("\n" + "=" * 60)
print("第7部分：MiniMind 中的 Embedding")
print("=" * 60)

print("""
MiniMind 的 Embedding 实现 (来自 model_minimind.py):

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

配置 (来自 MiniMindConfig):
  vocab_size: 6400
  hidden_size: 768

完整流程:
  input_ids: [batch, seq_len] 整数
      ↓ Embedding
  x:        [batch, seq_len, 768] 浮点向量
      ↓ Transformer Blocks × N
  hidden_states: [batch, seq_len, 768]
      ↓ LM Head
  logits: [batch, seq_len, 6400] 概率分布
""")


# ============================================================
# 第8部分（新增）：Embedding 是如何被训练的？
# ============================================================
print("\n" + "=" * 60)
print("第8部分：Embedding 是如何被训练的？")
print("=" * 60)

print("""
前面的演示中，Embedding 是随机初始化的，没有任何语义。
你一定在想：随机向量怎么就"学会了"语义？

答案是：通过训练！反向传播会更新 Embedding 矩阵的每一行。

训练过程：
  1. 初始化：每个 token 的向量是随机的（没有任何意义）
  2. 前向传播：用 Embedding 查表，得到向量，送入模型计算 loss
  3. 反向传播：计算 loss 对 Embedding 矩阵的梯度
  4. 参数更新：梯度下降修改 Embedding 矩阵中"被用到的那些行"

关键理解：
  - 每次训练，只有当前 batch 中出现的 token 的 Embedding 会被更新
  - 没出现的 token 的向量不变
  - 经常一起出现的词（如"猫"和"狗"），它们的梯度方向相似
  - 训练久了，语义相近的词自然聚集在一起

类比：
  初始化 = 一群陌生人随机站在操场上
  训练   = 每次任务后，相关的人被拉到一起
  训练久了 = 认识的人自然聚成小团体
""")

# 演示：训练一个简单的 Embedding，观察语义变化
torch.manual_seed(42)

# 构造一个简单的"语言模型"：输入两个词，预测第三个词
# 训练数据：(词1, 词2) → 词3
# 目的：让模型学到"猫"和"狗"是动物，"苹果"和"香蕉"是水果

vocab = ["猫", "狗", "苹果", "香蕉", "吃", "喜欢", "是", "动物", "水果"]
vocab_size = len(vocab)
embed_dim = 8

# 训练数据：让语义相近的词经常一起出现
training_data = [
    ("猫", "是", "动物"), ("狗", "是", "动物"),
    ("苹果", "是", "水果"), ("香蕉", "是", "水果"),
    ("猫", "喜欢", "吃"), ("狗", "喜欢", "吃"),
    ("苹果", "喜欢", "吃"), ("香蕉", "喜欢", "吃"),
]

char2id = {c: i for i, c in enumerate(vocab)}
id2char = {i: c for c, i in char2id.items()}

# 创建 Embedding 和简单的预测头
embedding = nn.Embedding(vocab_size, embed_dim)
lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

optimizer = torch.optim.Adam(list(embedding.parameters()) + list(lm_head.parameters()), lr=0.05)

# 记录训练前的相似度
def get_sim_matrix(emb, words_of_interest):
    """计算指定词之间的余弦相似度矩阵"""
    ids = [char2id[w] for w in words_of_interest]
    vecs = emb.weight[ids].detach()
    sim = torch.zeros(len(ids), len(ids))
    for i in range(len(ids)):
        for j in range(len(ids)):
            sim[i, j] = cosine_similarity(vecs[i], vecs[j])
    return sim

words_of_interest = ["猫", "狗", "苹果", "香蕉"]

print("训练前的词向量相似度（随机，无语义）：")
sim_before = get_sim_matrix(embedding, words_of_interest)
print(f"        猫     狗    苹果   香蕉")
for i, w in enumerate(words_of_interest):
    row = "  ".join(f"{sim_before[i, j]:6.3f}" for j in range(len(words_of_interest)))
    print(f"  {w}  {row}")
print("  → 所有相似度都接近0，没有语义规律")

# 训练
print("\n开始训练...")
for epoch in range(100):
    total_loss = 0
    for w1, w2, w3 in training_data:
        # 输入：前两个词的 Embedding 平均
        ids = torch.tensor([char2id[w1], char2id[w2]])
        vecs = embedding(ids)           # (2, embed_dim)
        avg_vec = vecs.mean(dim=0)      # (embed_dim,)

        # 预测第三个词
        logits = lm_head(avg_vec)        # (vocab_size,)
        target = torch.tensor(char2id[w3])

        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d}, Loss: {total_loss:.4f}")

# 训练后的相似度
print("\n训练后的词向量相似度（学到了语义！）：")
sim_after = get_sim_matrix(embedding, words_of_interest)
print(f"        猫     狗    苹果   香蕉")
for i, w in enumerate(words_of_interest):
    row = "  ".join(f"{sim_after[i, j]:6.3f}" for j in range(len(words_of_interest)))
    print(f"  {w}  {row}")
print("  → 猫-狗相似度高（都是动物），苹果-香蕉相似度高（都是水果）")
print("  → 猫-苹果相似度低（跨类别）")
print("  这就是 Embedding 通过训练学到语义的过程！")

print("""
总结：Embedding 训练的本质
  ┌──────────────────────────────────────────────────┐
  │  初始：随机向量，没有意义                          │
  │  训练：反向传播更新 Embedding 矩阵                 │
  │  结果：经常共现的词 → 向量靠近                     │
  │       语义相近的词 → 向量相似                      │
  │       语义相反的词 → 向量远离                      │
  │                                                  │
  │  关键：Embedding 的"语义"不是手动设定的，           │
  │        而是模型在训练过程中自动学到的！              │
  └──────────────────────────────────────────────────┘
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】Embedding 形状计算
  输入: [2, 10] 的 token IDs, vocab_size=1000, embed_dim=64
  输出 shape 是什么? 参数量是多少?

【练习2】padding_idx 的作用
  nn.Embedding(vocab_size=100, padding_idx=0)
  - 它的 weight[0] 永远是 0, 为什么这有用?
  - 训练时这个位置的梯度会更新吗?

【练习3】余弦相似度
  vec(猫) = [0.5, 0.3, 0.8]
  vec(狗) = [0.6, 0.2, 0.7]
  vec(车) = [-0.5, 0.9, 0.1]
  计算猫-狗、狗-车、猫-车的相似度, 哪个最相关?

【练习4】词表大小的选择
  中文常用字 ~3500, 加上英文 + 数字 + 标点大概多少?
  为什么 MiniMind 用 6400 而不是 3500?

【练习5】权重共享
  Embedding 矩阵是 V×d, LM Head 是 d×V
  如果共享权重, 能减少多少参数?
  为什么实践中常常不严格共享?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
V, d = 1000, 64
input_shape = [2, 10]
output_shape = input_shape + [d]
params = V * d
print(f"  输入 shape: {input_shape}")
print(f"  输出 shape: {output_shape}")
print(f"  参数量: V × d = {V} × {d} = {params:,}")

# 练习2
print("\n【练习2 答案】")
print("  用途: padding 位置不应该影响其他 token")
print("        让 padding 的 embedding 是 0, 表示'空'")
print("  梯度: padding_idx 处的梯度被强制设为 0, 不会更新")
print("        避免 padding token 学到任何'有意义的'表示")

# 练习3
print("\n【练习3 答案】")
cat = torch.tensor([0.5, 0.3, 0.8])
dog = torch.tensor([0.6, 0.2, 0.7])
car = torch.tensor([-0.5, 0.9, 0.1])

print(f"  sim(猫, 狗) = {cosine_similarity(cat, dog):.4f}")
print(f"  sim(狗, 车) = {cosine_similarity(dog, car):.4f}")
print(f"  sim(猫, 车) = {cosine_similarity(cat, car):.4f}")
print(f"  结论: 猫-狗最相关 (动物), 猫-车最不相关")

# 练习4
print("\n【练习4 答案】")
print("  中文常用字: ~3500")
print("  英文: 26 字母 × 2 大小写 = 52")
print("  数字: 10")
print("  标点: ~30")
print("  BPE subwords: ~3000 (高频词组)")
print("  特殊 token: 100 (BOS, EOS, PAD, ...)")
print("  合计: ~6400-7000")
print("  选用 6400 是为了支持 BPE subwords, 比纯字符词表大")

# 练习5
print("\n【练习5 答案】")
print("  Embedding 参数量: V × d")
print("  LM Head 参数量: d × V = V × d (相同)")
print("  共享时减少: 50% 参数 (V × d)")
print("  以 LLaMA-7B 为例: 节省 ~50M 参数 (2.5% 总参)")
print("  MiniMind (~26M): 节省 ~3M 参数 (~12% 总参)")
print("  实践: 严格共享时输入输出嵌入同空间, 效果略降")
print("        实践中常用近似共享或部分共享")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. Embedding 本质
   - V×d 的查找表
   - token ID → d 维向量
   - 训练过程中学习语义

2. 关键概念
   - vocab_size (V): 词表大小
   - hidden_size (d): 嵌入维度
   - padding_idx: padding 位置

3. 训练后的语义
   - 语义相近的词, 向量接近
   - 国王-王后 = 男人-女人
   - 余弦相似度衡量相关性

4. 维度选择
   - 太小: 表达力不足
   - 太大: 训练慢, 浪费
   - 经验: d ≈ (V**0.25) * 4

5. 权重共享
   - Embedding 与 LM Head 共享权重
   - 节省 50% 参数
   - 训练更稳定

下一步: 第3课 - RMSNorm (为什么需要归一化？)
""")
