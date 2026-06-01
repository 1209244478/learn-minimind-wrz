"""
第1课：Tokenizer — 文本如何变成数字？
=====================================

为什么模型需要 Tokenizer？
  计算机只懂数字, 不懂文字
  所有 AI 模型 (LLM、CV、Speech) 都需要把输入转为数字

Tokenizer 做了什么？
  文字 "你好世界"  →  数字序列 [1234, 5678, 9012, 3456]

三种主流分词方式:

  1. 字符级 (Char-level)
     原理: 每个字符一个 token
     例子: "你好" → ['你', '好'] → [45, 67]
     优点: 词表小 (~100 个 token)
     缺点: 序列很长, 训练慢

  2. 词级 (Word-level)
     原理: 每个词一个 token
     例子: "你好世界" → ['你好', '世界'] → [1234, 5678]
     优点: 语义清晰
     缺点: 词表爆炸 (几十万), OOV 问题

  3. 子词级 (Subword, BPE)
     原理: 高频词组合并, 低频词拆分
     例子: "unhappiness" → ['un', 'happiness'] → [123, 456]
     优点: 平衡了词表大小和序列长度
     缺点: 实现复杂

类比:
  Tokenizer 就像一本"密码本"
  - 中文: 字符 → 数字
  - 英文: 单词 → 数字
  - 模型只能看数字, 看不到原文字

运行: python lessons/lesson01_tokenizer.py
"""

import torch
import re
from collections import Counter, defaultdict


# ============================================================
# 第1部分：字符级 Tokenizer (最简单)
# ============================================================
print("=" * 60)
print("第1部分：字符级 Tokenizer")
print("=" * 60)


class CharTokenizer:
    """字符级分词器 - 每个字符一个 token"""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return ''.join(self.id_to_char[i] for i in ids)


# 实验
sample = "hello world 你好世界"
tokenizer = CharTokenizer(sample)
ids = tokenizer.encode(sample)
print(f"\n原文: '{sample}'")
print(f"编码: {ids}")
print(f"词表大小: {tokenizer.vocab_size}")
print(f"序列长度: {len(ids)}")
print(f"解码: '{tokenizer.decode(ids)}'")

# 优缺点演示
print("\n字符级 Tokenizer 优缺点:")
print("  ✓ 词表小 (几十个字符)")
print("  ✗ 序列长 (每个字符 1 个 token)")
print(f"  例: 100 字文本 → {100} tokens, 10000 字文本 → {10000} tokens")


# ============================================================
# 第2部分：词级 Tokenizer
# ============================================================
print("\n" + "=" * 60)
print("第2部分：词级 Tokenizer")
print("=" * 60)


class WordTokenizer:
    """词级分词器 - 每个词一个 token"""

    def __init__(self, text):
        # 简单的分词: 按空格和标点分割
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        self.words = sorted(set(words))
        self.vocab_size = len(self.words)
        self.word_to_id = {w: i for i, w in enumerate(self.words)}
        self.id_to_word = {i: w for i, w in enumerate(self.words)}

    def _tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

    def encode(self, text):
        words = self._tokenize(text)
        return [self.word_to_id.get(w, -1) for w in words]  # -1 = UNK

    def decode(self, ids):
        return ' '.join(self.id_to_word.get(i, '<UNK>') for i in ids)


# 准备训练文本
corpus = """
The quick brown fox jumps over the lazy dog.
The dog barks at the fox.
The fox runs away quickly.
机器学习是人工智能的一个分支。
深度学习是机器学习的一个子集。
神经网络是深度学习的核心。
"""

word_tokenizer = WordTokenizer(corpus)
print(f"\n词表大小: {word_tokenizer.vocab_size}")
print(f"前 20 个词: {word_tokenizer.words[:20]}")

# 测试
test = "The fox is happy"
ids = word_tokenizer.encode(test)
print(f"\n原文: '{test}'")
print(f"分词: {word_tokenizer._tokenize(test)}")
print(f"编码: {ids}")
print(f"解码: '{word_tokenizer.decode(ids)}'")

print("\n词级 Tokenizer 优缺点:")
print("  ✓ 语义清晰")
print("  ✗ 词表大 (几十万)")
print("  ✗ OOV 问题 (没见过的词 → UNK)")
print("  ✗ 形态变化: run/runs/running 各算一个词")


# ============================================================
# 第3部分：BPE (Byte Pair Encoding) 算法
# ============================================================
print("\n" + "=" * 60)
print("第3部分：BPE 算法")
print("=" * 60)

print("""
BPE 是 GPT、Llama、MiniMind 等所有主流 LLM 用的方法。
核心思想: 从字符开始, 不断合并最高频的相邻对

算法流程:
  1. 初始化词表 = 所有单字符
  2. 统计相邻字符对的出现频率
  3. 找到最高频的字符对 (a, b), 合并为新 token "ab"
  4. 把 "ab" 加入词表
  5. 重复 2-4, 直到词表大小达到目标
""")


def get_pairs(word):
    """获取词中所有相邻字符对"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def bpe_train(corpus, num_merges=10, end_of_word='</w>'):
    """BPE 训练算法"""
    # 1. 初始化: 每个词拆分为字符
    word_freqs = Counter(corpus.split())
    vocab = set()
    for word in word_freqs:
        vocab.update(list(word))
    vocab.add(end_of_word)

    # 每个词表示为字符列表
    splits = {word: [c for c in word] + [end_of_word] for word in word_freqs}

    print(f"初始词表: {sorted(vocab)[:20]}... (共 {len(vocab)} 个)")

    # 2-N. 迭代合并
    merges = []
    for i in range(num_merges):
        # 统计相邻对频率
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            splits_word = splits[word]
            for pair in get_pairs(splits_word):
                pair_freqs[pair] += freq

        if not pair_freqs:
            break

        # 找最高频对
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]

        # 合并
        new_token = ''.join(best_pair)
        merges.append((best_pair, new_token))
        vocab.add(new_token)

        # 更新所有词的拆分
        new_splits = {}
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == best_pair:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        splits = new_splits

        print(f"  合并 #{i+1}: {best_pair} -> '{new_token}' (频率: {best_freq})")

    return vocab, merges, splits


# 准备语料
bpe_corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider"
print(f"\n训练语料: '{bpe_corpus}'")

vocab, merges, splits = bpe_train(bpe_corpus, num_merges=8)

print(f"\n最终词表大小: {len(vocab)}")
print(f"学习到的合并规则: {merges}")


# ============================================================
# 第4部分：BPE 编码/解码
# ============================================================
print("\n" + "=" * 60)
print("第4部分：BPE 编码/解码")
print("=" * 60)


class SimpleBPE:
    """简化的 BPE 分词器"""

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {t: i for i, t in enumerate(sorted(vocab))}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def _tokenize_word(self, word):
        """对单个词应用 BPE"""
        split = list(word) + ['</w>']
        for pair, new_token in self.merges:
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == pair:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split
        return split

    def encode(self, text):
        tokens = []
        for word in text.split():
            tokens.extend(self._tokenize_word(word))
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, ids):
        text = ''.join(self.id_to_token[i] for i in ids)
        text = text.replace('</w>', ' ')
        return text.strip()


bpe = SimpleBPE(vocab, merges)
test = "low newer"
ids = bpe.encode(test)
print(f"\n原文: '{test}'")
print(f"编码: {ids}")
print(f"对应 token: {[bpe.id_to_token[i] for i in ids]}")
print(f"解码: '{bpe.decode(ids)}'")


# ============================================================
# 第5部分：完整对比
# ============================================================
print("\n" + "=" * 60)
print("第5部分：三种 Tokenizer 对比")
print("=" * 60)

test_text = "The quick brown fox jumps over the lazy dog"
print(f"\n测试文本: '{test_text}' ({len(test_text)} 字符)")

# 字符级
char_ids = CharTokenizer(test_text).encode(test_text)
print(f"\n字符级: {len(char_ids)} tokens")
print(f"  前 10 个: {char_ids[:10]}")

# 词级
word_ids = WordTokenizer(corpus + " " + test_text).encode(test_text)
print(f"\n词级: {len(word_ids)} tokens")
print(f"  编码: {word_ids}")

# 总结
print("\n对比表:")
print("┌──────────────┬──────────┬──────────┬──────────┐")
print("│ 方案          │ 词表大小  │ 序列长度  │ OOV 处理  │")
print("├──────────────┼──────────┼──────────┼──────────┤")
print("│ 字符级        │ ~100     │ 长       │ 无       │")
print("│ 词级          │ 10万+    │ 短       │ UNK      │")
print("│ BPE          │ 1-10万   │ 中       │ 字符回退  │")
print("└──────────────┴──────────┴──────────┴──────────┘")


# ============================================================
# 第6部分：MiniMind 的 Tokenizer
# ============================================================
print("\n" + "=" * 60)
print("第6部分：MiniMind 实际用的 Tokenizer")
print("=" * 60)

print("""
MiniMind 使用预训练的 BPE 分词器, 词表大小 ~6400

实际使用:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("minimind")

# 编码
text = "今天天气真好"
ids = tokenizer.encode(text)
print(ids)        # [101, 234, 567, 890, 102]

# 解码
text2 = tokenizer.decode(ids)
print(text2)      # "今天天气真好"

# 批量处理
batch = tokenizer(["你好", "再见"], padding=True, return_tensors="pt")
```

MiniMind 的特点:
  - 词表小 (~6400), 训练快
  - 中文 + 英文 + 数字 + 标点
  - 用 BPE, 平衡词表和效率
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】字符统计
  给定文本 "Hello, World! 你好，世界！"
  统计有多少个不同字符, 用 CharTokenizer 编码后序列长度是多少

【练习2】词级 OOV 问题
  词表只包含 ["我", "爱", "学习"]
  编码 "我爱编程" 会得到什么结果？
  这是什么问题的体现？

【练习3】BPE 合并次数
  如果我们想要词表大小为 100, 初始字符有 50 个, 需要多少次合并？

【练习4】BPE 的优势
  假设有 100 万个不同英文单词, 用 BPE 词表大小 32000
  相比纯词级, 减少了多少词表大小？增加了多少序列长度？

【练习5】实现一个简化版 BPE
  输入: "a a a a b b c"
  目标: 3 次合并, 写出每次合并的 pair
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
text1 = "Hello, World! 你好，世界！"
ct = CharTokenizer(text1)
ids1 = ct.encode(text1)
print(f"  文本: '{text1}'")
print(f"  不同字符数: {ct.vocab_size}")
print(f"  编码后序列长度: {len(ids1)}")
print(f"  编码: {ids1}")

# 练习2
print("\n【练习2 答案】")
print("  编码 '我爱编程': ['我', '爱', '编程']")
print("  词表中没有 '编程' → 编码为 [-1] 或 <UNK>")
print("  这是 OOV (Out-Of-Vocabulary) 问题的体现")

# 练习3
print("\n【练习3 答案】")
print("  词表目标: 100")
print("  初始: 50")
print("  每次合并 +1 个 token")
print("  合并次数: 100 - 50 = 50 次")

# 练习4
print("\n【练习4 答案】")
print("  纯词级: 100万 词")
print("  BPE: 32000 词 (减少 96.8%)")
print("  代价: 平均每个单词被拆成 1.5-2 个 subword")
print("  序列长度增加 50-100%, 但词表缩小 30 倍")

# 练习5
print("\n【练习5 答案】")
text5 = "a a a a b b c"
print(f"  文本: '{text5}'")
print(f"  初始: ['a', 'a', 'a', 'a', 'b', 'b', 'c']")
print(f"  相邻对频率: ('a','a')=3, ('a','b')=1, ('b','b')=1, ('b','c')=1")
print()
print(f"  合并 1: ('a', 'a') 频率 3 → 'aa'")
print(f"    序列变为: ['aa', 'a', 'a', 'b', 'b', 'c']")
print(f"    相邻对频率: ('aa','a')=1, ('a','a')=1, ('a','b')=1, ('b','b')=1, ('b','c')=1")
print()
print(f"  合并 2: ('b', 'b') 频率 1 → 'bb'  (多个频率=1, 任选一个)")
print(f"    序列变为: ['aa', 'a', 'a', 'bb', 'c']")
print(f"    相邻对频率: ('aa','a')=1, ('a','a')=1, ('a','bb')=1, ('bb','c')=1")
print()
print(f"  合并 3: ('aa', 'a') 频率 1 → 'aaa'  (同样任选)")
print(f"    序列变为: ['aaa', 'a', 'bb', 'c']")
print(f"    最终词表: a, b, c, aa, bb, aaa")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. Tokenizer 作用
   - 把文字转为数字, 模型才能处理
   - 词表大小 vs 序列长度的权衡

2. 三种方案
   - 字符级: 词表小, 序列长
   - 词级: 语义清晰, 词表爆炸
   - BPE: 平衡方案, 所有现代 LLM 的选择

3. BPE 原理
   - 从字符开始, 不断合并高频相邻对
   - 平衡词表大小和序列长度
   - 没有 OOV 问题 (可以回退到字符)

4. MiniMind 用 BPE
   - 词表大小 ~6400
   - 训练快, 推理快
   - 支持中英文

5. 实际应用
   - 使用 transformers 库的 AutoTokenizer
   - encode/decode 一行代码搞定

下一步: 第2课 - Embedding (数字如何变成向量？)
""")
