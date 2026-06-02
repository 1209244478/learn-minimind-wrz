# ============================================================================
# 第25课：数据准备与清洗 — LLM 训练的基石
# ============================================================================
"""
"Garbage in, garbage out" — 数据质量决定模型质量。
本课介绍 LLM 训练中数据准备的全流程。

本课内容：
1. 预训练数据：网页抓取与清洗
2. SFT 数据：指令数据的构造与格式
3. DPO 数据：偏好数据的收集
4. 数据质量控制：去重、过滤、去污染
5. Tokenizer 训练

关键概念：
- 数据清洗 (Data Cleaning)：去除低质量、重复、有害数据
- 数据去重 (Deduplication)：删除重复或近似重复的文本
- 数据去污染 (Decontamination)：移除与评估集重叠的数据
- 数据混合 (Data Mixing)：不同类型数据的配比
"""

import torch
import re
import json
import hashlib
from collections import Counter

# ============================================================================
# 1. 预训练数据
# ============================================================================
"""
预训练数据的来源：
  - Common Crawl：网页抓取数据（最大来源，但质量参差不齐）
  - Wikipedia：高质量百科数据
  - Books：书籍语料
  - GitHub：代码数据
  - ArXiv：学术论文
  - StackExchange：问答数据

数据量级：
  - GPT-3: 300B tokens
  - LLaMA: 1.4T tokens
  - MiniMind: ~1B tokens（教学用途）

数据处理流水线：
  原始数据 → 语言过滤 → 质量过滤 → 去重 → 去污染 → 最终数据
"""

print("=" * 70)
print("第25课：数据准备与清洗")
print("=" * 70)

# ============================================================================
# 2. 数据清洗
# ============================================================================


def clean_text(text):
    """基础文本清洗

    处理步骤：
    1. 去除 HTML 标签
    2. 规范化空白字符
    3. 去除特殊字符
    4. 去除过短的行
    """
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除 URL
    text = re.sub(r'https?://\S+', '', text)

    # 规范化空白字符
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空白
    text = text.strip()

    return text


def filter_quality(text, min_length=50, max_length=100000,
                   min_avg_word_length=3, max_avg_word_length=10):
    """质量过滤

    过滤条件：
    1. 文本长度在合理范围
    2. 平均词长在合理范围
    3. 特殊字符比例不过高
    """
    # 长度过滤
    if len(text) < min_length or len(text) > max_length:
        return False

    # 平均词长过滤（太短=无意义，太长=乱码）
    words = text.split()
    if len(words) == 0:
        return False
    avg_word_length = sum(len(w) for w in words) / len(words)
    if avg_word_length < min_avg_word_length or avg_word_length > max_avg_word_length:
        return False

    # 特殊字符比例过滤
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / len(text)
    if special_ratio > 0.3:
        return False

    return True


# 演示数据清洗
print("\n--- 数据清洗示例 ---")
raw_texts = [
    "<html><body>这是正常的文本内容，包含足够的信息量。</body></html>",
    "a b c d",  # 太短
    "asdfghjklqwertyuiopzxcvbnm",  # 无意义
    "正常的中等长度文本，包含有意义的词汇和句子结构，适合用于训练语言模型。",
    "!!!@@@###$$$%%%",  # 特殊字符过多
]

for text in raw_texts:
    cleaned = clean_text(text)
    passed = filter_quality(cleaned)
    status = "✓ 保留" if passed else "✗ 过滤"
    print(f"  {status}: {cleaned[:50]}...")

# ============================================================================
# 3. 数据去重
# ============================================================================


def exact_dedup(texts):
    """精确去重：删除完全相同的文本"""
    seen = set()
    unique = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def minhash_dedup(texts, n_grams=5, num_hashes=128, threshold=0.7):
    """MinHash 近似去重

    原理：
    1. 将文本分解为 n-gram 集合
    2. 用多个哈希函数计算 MinHash 签名
    3. 比较签名相似度，超过阈值的视为重复

    这里用简化版本演示原理。
    """
    def get_ngrams(text, n):
        tokens = text.split()
        return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def minhash_signature(ngrams, num_hashes):
        signature = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for ng in ngrams:
                h = int(hashlib.md5(f"{ng}_{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def jaccard_similarity(sig1, sig2):
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    signatures = []
    for text in texts:
        ngrams = get_ngrams(text, n_grams)
        if ngrams:
            sig = minhash_signature(ngrams, num_hashes)
        else:
            sig = [0] * num_hashes
        signatures.append(sig)

    # 找出重复项
    unique_indices = []
    for i in range(len(texts)):
        is_duplicate = False
        for j in unique_indices:
            sim = jaccard_similarity(signatures[i], signatures[j])
            if sim > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)

    return [texts[i] for i in unique_indices]


# 演示去重
print("\n--- 数据去重示例 ---")
texts_with_dup = [
    "人工智能是计算机科学的一个分支",
    "人工智能是计算机科学的一个分支",  # 完全重复
    "机器学习是人工智能的一个子领域",
    "机器学习是AI的一个子领域",  # 近似重复
    "深度学习使用多层神经网络",
]

exact_unique = exact_dedup(texts_with_dup)
print(f"精确去重: {len(texts_with_dup)} → {len(exact_unique)} 条")

approx_unique = minhash_dedup(texts_with_dup, threshold=0.5)
print(f"近似去重: {len(texts_with_dup)} → {len(approx_unique)} 条")

# ============================================================================
# 4. SFT 数据格式
# ============================================================================
"""
SFT 数据的标准格式（JSONL）：

{"conversations": [
  {"role": "system", "content": "你是一个有用的助手"},
  {"role": "user", "content": "什么是机器学习？"},
  {"role": "assistant", "content": "机器学习是AI的子集..."}
]}

数据来源：
1. 人工编写（质量最高，成本最高）
2. GPT-4 生成（Self-Instruct 方法）
3. 开源数据集（Alpaca, ShareGPT 等）
4. 现有数据集改写

数据质量标准：
- 回答准确、有帮助
- 格式规范、语言流畅
- 无有害内容
- 多样性足够
"""


def validate_sft_data(sample):
    """验证 SFT 数据质量

    Args:
        sample: 一条 SFT 数据

    Returns:
        (is_valid, reason)
    """
    # 检查必需字段
    if "conversations" not in sample:
        return False, "缺少 conversations 字段"

    convs = sample["conversations"]

    # 检查至少有一轮 user-assistant 对话
    has_user = any(c["role"] == "user" for c in convs)
    has_assistant = any(c["role"] == "assistant" for c in convs)
    if not (has_user and has_assistant):
        return False, "缺少 user 或 assistant 角色"

    # 检查内容不为空
    for c in convs:
        if not c.get("content", "").strip():
            return False, f"角色 {c['role']} 的内容为空"

    # 检查 assistant 回答长度合理
    for c in convs:
        if c["role"] == "assistant":
            if len(c["content"]) < 5:
                return False, "assistant 回答过短"
            if len(c["content"]) > 5000:
                return False, "assistant 回答过长"

    return True, "OK"


# 演示 SFT 数据验证
print("\n--- SFT 数据验证示例 ---")
sft_samples = [
    {"conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
    ]},
    {"conversations": [
        {"role": "user", "content": "什么是AI？"},
        {"role": "assistant", "content": ""}
    ]},
    {"conversations": [
        {"role": "user", "content": "介绍Python"},
        {"role": "assistant", "content": "Python是一种高级编程语言，以简洁易读著称。它广泛应用于Web开发、数据科学、人工智能等领域。"}
    ]},
]

for i, sample in enumerate(sft_samples):
    valid, reason = validate_sft_data(sample)
    status = "✓" if valid else "✗"
    print(f"  样本{i + 1} {status}: {reason}")

# ============================================================================
# 5. DPO 数据格式
# ============================================================================
"""
DPO 数据格式（JSONL）：

{
  "chosen": [
    {"role": "user", "content": "什么是AI？"},
    {"role": "assistant", "content": "AI是人工智能的缩写，是计算机科学的分支..."}
  ],
  "rejected": [
    {"role": "user", "content": "什么是AI？"},
    {"role": "assistant", "content": "AI就是很厉害的东西。"}
  ]
}

数据来源：
1. 人工标注（给同一问题写好/差回答）
2. 模型对比（用不同模型生成，人类选更好的）
3. 自动构造（用规则判断回答质量）
"""


def create_dpo_pair(prompt, good_response, bad_response):
    """创建一条 DPO 训练数据"""
    return {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": good_response}
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": bad_response}
        ]
    }


# 演示 DPO 数据构造
print("\n--- DPO 数据构造示例 ---")
dpo_pair = create_dpo_pair(
    prompt="解释什么是深度学习",
    good_response="深度学习是机器学习的一个子集，使用多层神经网络自动学习数据的层次化表示。它在图像识别、自然语言处理等领域取得了突破性进展。",
    bad_response="深度学习就是学得很深。"
)
print(f"Chosen: {dpo_pair['chosen'][1]['content'][:50]}...")
print(f"Rejected: {dpo_pair['rejected'][1]['content']}")

# ============================================================================
# 6. 数据去污染
# ============================================================================
"""
去污染 (Decontamination)：确保训练数据不包含评估集的内容。

为什么重要？
  如果训练数据包含评估集的题目和答案，
  模型会"记住"答案，评估分数虚高，不能反映真实能力。

方法：
  1. N-gram 匹配：检查训练数据是否包含评估集的 n-gram
  2. MinHash：近似匹配，检测改写后的评估数据
  3. 人工审核：对可疑数据进行人工检查
"""


def decontaminate(train_texts, eval_texts, n_gram_size=13):
    """简单的 n-gram 去污染

    检查训练数据是否包含评估集的 n-gram
    """
    # 收集评估集的 n-grams
    eval_ngrams = set()
    for text in eval_texts:
        tokens = text.split()
        for i in range(len(tokens) - n_gram_size + 1):
            ngram = ' '.join(tokens[i:i + n_gram_size])
            eval_ngrams.add(ngram)

    # 过滤训练数据
    clean_texts = []
    contaminated = 0
    for text in train_texts:
        tokens = text.split()
        has_contamination = False
        for i in range(len(tokens) - n_gram_size + 1):
            ngram = ' '.join(tokens[i:i + n_gram_size])
            if ngram in eval_ngrams:
                has_contamination = True
                break
        if not has_contamination:
            clean_texts.append(text)
        else:
            contaminated += 1

    return clean_texts, contaminated


# 演示去污染
print("\n--- 数据去污染示例 ---")
eval_data = [
    "中国的首都是北京，这是地理常识",
]
train_data = [
    "北京是中国的首都，拥有悠久的历史",  # 有重叠
    "机器学习是人工智能的重要分支",  # 无重叠
]

clean, n_contaminated = decontaminate(train_data, eval_data, n_gram_size=5)
print(f"训练数据: {len(train_data)} 条")
print(f"污染数据: {n_contaminated} 条")
print(f"清洗后: {len(clean)} 条")

# ============================================================================
# 7. 数据混合策略
# ============================================================================
"""
不同类型数据的混合比例对模型效果影响很大。

典型的数据混合：
┌──────────────┬────────────┬──────────────────────────┐
│ 数据类型     │ 比例       │ 说明                     │
├──────────────┼────────────┼──────────────────────────┤
│ 网页文本     │ 60-70%     │ 通用知识                 │
│ 代码         │ 10-15%     │ 推理能力                 │
│ 书籍         │ 5-10%      │ 深度知识                 │
│ 学术论文     │ 5%         │ 专业知识                 │
│ 百科         │ 5%         │ 事实知识                 │
│ 问答         │ 5%         │ 对话能力                 │
└──────────────┴────────────┴──────────────────────────┘

SFT 数据混合：
┌──────────────┬────────────┬──────────────────────────┐
│ 数据类型     │ 比例       │ 说明                     │
├──────────────┼────────────┼──────────────────────────┤
│ 通用对话     │ 40%        │ 日常对话能力             │
│ 代码生成     │ 20%        │ 编程能力                 │
│ 数学推理     │ 15%        │ 逻辑推理                 │
│ 知识问答     │ 15%        │ 事实知识                 │
│ 写作创作     │ 10%        │ 创意写作                 │
└──────────────┴────────────┴──────────────────────────┘
"""

print("\n--- 数据混合策略 ---")
data_mix = {
    "网页文本": 0.65,
    "代码": 0.12,
    "书籍": 0.08,
    "学术论文": 0.05,
    "百科": 0.05,
    "问答": 0.05,
}
print("预训练数据混合:")
for dtype, ratio in data_mix.items():
    bar = "█" * int(ratio * 40)
    print(f"  {dtype:8s}: {ratio:.0%} {bar}")

# ============================================================================
# 8. Tokenizer 训练
# ============================================================================
"""
Tokenizer 的选择对模型效果影响很大：

1. BPE (Byte Pair Encoding)
   - GPT 系列使用
   - 从字符级开始，逐步合并高频字节对
   - 适合英文

2. SentencePiece / BPE
   - LLaMA 系列使用
   - 支持多语言
   - 中文通常用 32K-128K 词表

3. WordPiece
   - BERT 使用
   - 类似 BPE，但选择方式不同

词表大小的影响：
  - 太小（<10K）：编码效率低，序列太长
  - 太大（>200K）：嵌入矩阵太大，参数浪费
  - 中文推荐：32K-64K
  - 多语言推荐：128K-256K
"""

print("\n--- Tokenizer 词表大小影响 ---")
for vocab_size in [10000, 32000, 64000, 128000, 256000]:
    # 假设 dim=512
    dim = 512
    emb_params = vocab_size * dim / 1e6  # 百万
    print(f"  词表={vocab_size:>6d}, dim={dim} → 嵌入层参数={emb_params:.1f}M")

# ============================================================================
# 练习
# ============================================================================
print("\n" + "=" * 70)
print("练习")
print("=" * 70)

# 练习1：数据清洗
print("\n练习1：数据清洗")
dirty_text = "<p>这是HTML文本&nbsp;&nbsp;包含多余空格和标签</p>"
cleaned = clean_text(dirty_text)
print(f"原始: {dirty_text}")
print(f"清洗: {cleaned}")

# 练习2：质量过滤
print("\n练习2：质量过滤")
test_texts = [
    "这是一段正常的文本，长度适中，内容有意义。",
    "短",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
for text in test_texts:
    passed = filter_quality(text, min_length=10)
    print(f"  {'✓' if passed else '✗'}: {text[:30]}...")

# 练习3：精确去重
print("\n练习3：精确去重")
dup_texts = ["你好世界", "你好世界", "Hello World", "你好世界", "Hello World"]
unique = exact_dedup(dup_texts)
print(f"原始: {len(dup_texts)} 条")
print(f"去重后: {len(unique)} 条 → {unique}")

# 练习4：SFT 数据验证
print("\n练习4：SFT 数据验证")
test_sample = {
    "conversations": [
        {"role": "user", "content": "什么是Python？"},
        {"role": "assistant", "content": "Python是一种广泛使用的高级编程语言。"}
    ]
}
valid, reason = validate_sft_data(test_sample)
print(f"验证结果: {'✓ 有效' if valid else '✗ 无效'} - {reason}")

print("\n" + "=" * 70)
print("第25课总结：")
print("  1. 数据质量决定模型质量：Garbage in, garbage out")
print("  2. 数据清洗：去HTML、规范化、质量过滤")
print("  3. 数据去重：精确去重 + MinHash 近似去重")
print("  4. 数据去污染：确保训练数据不包含评估集内容")
print("  5. 数据混合：不同类型数据的配比影响模型能力")
print("  6. Tokenizer：词表大小影响编码效率和参数量")
print("=" * 70)
