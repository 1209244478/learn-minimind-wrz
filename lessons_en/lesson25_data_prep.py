# ============================================================================
# Lesson 25: Data Preparation & Cleaning — The Foundation of LLM Training
# ============================================================================
"""
"Garbage in, garbage out" — Data quality determines model quality.
This lesson covers the full data preparation pipeline for LLM training.

Topics:
1. Pretraining Data: Web Scraping & Cleaning
2. SFT Data: Instruction Data Construction & Format
3. DPO Data: Preference Data Collection
4. Data Quality Control: Dedup, Filter, Decontamination
5. Tokenizer Training

Key Concepts:
- Data Cleaning: Remove low-quality, duplicate, harmful data
- Deduplication: Delete exact or near-duplicate texts
- Decontamination: Remove data overlapping with evaluation sets
- Data Mixing: Proportions of different data types
"""

import torch
import re
import json
import hashlib
from collections import Counter

# ============================================================================
# 1. Pretraining Data
# ============================================================================
"""
Pretraining data sources:
  - Common Crawl: Web crawl data (largest source, mixed quality)
  - Wikipedia: High-quality encyclopedia data
  - Books: Book corpora
  - GitHub: Code data
  - ArXiv: Academic papers
  - StackExchange: Q&A data

Data scale:
  - GPT-3: 300B tokens
  - LLaMA: 1.4T tokens
  - MiniMind: ~1B tokens (for education)

Data processing pipeline:
  Raw data → Language filter → Quality filter → Dedup → Decontaminate → Final data
"""

print("=" * 70)
print("Lesson 25: Data Preparation & Cleaning")
print("=" * 70)

# [NEW] Real data sample showcase
print("\n" + "-" * 50)
print("[Three Types of Training Data — Real Samples]")
print("-" * 50)
print("""
1. Pre-training data (plain text, model learns "what word comes next"):
   ─────────────────────────────────────────
   {"text": "Natural language processing is an important direction in the field of artificial intelligence. It studies how computers can understand and generate human language. Applications include machine translation, text summarization, question answering systems, etc. In recent years, large language models based on Transformers have achieved breakthrough progress in NLP."}
   ─────────────────────────────────────────
   Feature: Plain text, no Q&A format, model learns statistical patterns of language

2. SFT data (instruction-response format, model learns "follow instructions"):
   ─────────────────────────────────────────
   {"messages": [
     {"role": "system", "content": "You are a helpful AI assistant."},
     {"role": "user", "content": "Please explain what gradient descent is"},
     {"role": "assistant", "content": "Gradient descent is an optimization algorithm used to minimize a loss function. It updates parameters along the negative gradient direction of the loss function, taking one step at a time, gradually approaching the optimal solution. Formula: θ = θ - α·∇L(θ), where α is the learning rate and ∇L is the gradient."}
   ]}
   ─────────────────────────────────────────
   Feature: Has clear "instruction" and "response", model learns to follow instructions

3. DPO data (preference pairs, model learns "what is a good response"):
   ─────────────────────────────────────────
   {"prompt": "How to learn programming?",
    "chosen": "I recommend starting with Python. It has clean syntax and rich community resources. Start with the official tutorial, learn while practicing, and gradually master basic concepts before trying small projects.",
    "rejected": "Programming is hard, I don't recommend learning it."}
   ─────────────────────────────────────────
   Feature: Same question has good and bad responses, model learns to prefer the good one
""")

# ============================================================================
# 2. Data Cleaning
# ============================================================================


def clean_text(text):
    """Basic text cleaning

    Steps:
    1. Remove HTML tags
    2. Normalize whitespace
    3. Remove special characters
    4. Remove very short lines
    """
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def filter_quality(text, min_length=50, max_length=100000,
                   min_avg_word_length=3, max_avg_word_length=10):
    """Quality filtering

    Filter criteria:
    1. Text length in reasonable range
    2. Average word length in reasonable range
    3. Special character ratio not too high
    """
    if len(text) < min_length or len(text) > max_length:
        return False

    words = text.split()
    if len(words) == 0:
        return False
    avg_word_length = sum(len(w) for w in words) / len(words)
    if avg_word_length < min_avg_word_length or avg_word_length > max_avg_word_length:
        return False

    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / len(text)
    if special_ratio > 0.3:
        return False

    return True


# Demo data cleaning
print("\n--- Data Cleaning Example ---")
raw_texts = [
    "<html><body>This is normal text with enough information content.</body></html>",
    "a b c d",  # Too short
    "asdfghjklqwertyuiopzxcvbnm",  # Nonsense
    "Normal medium-length text with meaningful vocabulary and sentence structure, suitable for training language models.",
    "!!!@@@###$$$%%%",  # Too many special chars
]

for text in raw_texts:
    cleaned = clean_text(text)
    passed = filter_quality(cleaned)
    status = "KEEP" if passed else "FILTER"
    print(f"  {status}: {cleaned[:50]}...")

# ============================================================================
# 3. Data Deduplication
# ============================================================================


def exact_dedup(texts):
    """Exact deduplication: Remove identical texts"""
    seen = set()
    unique = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def minhash_dedup(texts, n_grams=5, num_hashes=128, threshold=0.7):
    """MinHash approximate deduplication

    Principle:
    1. Break text into n-gram sets
    2. Compute MinHash signatures with multiple hash functions
    3. Compare signature similarity, above threshold = duplicate

    This is a simplified version for demonstration.
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


# Demo deduplication
print("\n--- Data Deduplication Example ---")
texts_with_dup = [
    "Artificial intelligence is a branch of computer science",
    "Artificial intelligence is a branch of computer science",  # Exact duplicate
    "Machine learning is a subfield of artificial intelligence",
    "Machine learning is a subfield of AI",  # Near duplicate
    "Deep learning uses multi-layer neural networks",
]

exact_unique = exact_dedup(texts_with_dup)
print(f"Exact dedup: {len(texts_with_dup)} → {len(exact_unique)} items")

approx_unique = minhash_dedup(texts_with_dup, threshold=0.5)
print(f"Approx dedup: {len(texts_with_dup)} → {len(approx_unique)} items")

# ============================================================================
# 4. SFT Data Format
# ============================================================================
"""
SFT data standard format (JSONL):

{"conversations": [
  {"role": "system", "content": "You are a helpful assistant"},
  {"role": "user", "content": "What is machine learning?"},
  {"role": "assistant", "content": "Machine learning is a subset of AI..."}
]}

Data sources:
1. Human-written (highest quality, highest cost)
2. GPT-4 generated (Self-Instruct method)
3. Open-source datasets (Alpaca, ShareGPT, etc.)
4. Existing dataset rewrites
"""


def validate_sft_data(sample):
    """Validate SFT data quality

    Returns: (is_valid, reason)
    """
    if "conversations" not in sample:
        return False, "Missing conversations field"

    convs = sample["conversations"]

    has_user = any(c["role"] == "user" for c in convs)
    has_assistant = any(c["role"] == "assistant" for c in convs)
    if not (has_user and has_assistant):
        return False, "Missing user or assistant role"

    for c in convs:
        if not c.get("content", "").strip():
            return False, f"Role {c['role']} has empty content"

    for c in convs:
        if c["role"] == "assistant":
            if len(c["content"]) < 5:
                return False, "Assistant response too short"
            if len(c["content"]) > 5000:
                return False, "Assistant response too long"

    return True, "OK"


# Demo SFT data validation
print("\n--- SFT Data Validation Example ---")
sft_samples = [
    {"conversations": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]},
    {"conversations": [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": ""}
    ]},
    {"conversations": [
        {"role": "user", "content": "Introduce Python"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, data science, and AI."}
    ]},
]

for i, sample in enumerate(sft_samples):
    valid, reason = validate_sft_data(sample)
    status = "VALID" if valid else "INVALID"
    print(f"  Sample {i + 1} {status}: {reason}")

# ============================================================================
# 5. DPO Data Format
# ============================================================================
"""
DPO data format (JSONL):

{
  "chosen": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence, a branch of computer science..."}
  ],
  "rejected": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is just something very powerful."}
  ]
}
"""


def create_dpo_pair(prompt, good_response, bad_response):
    """Create a DPO training data entry"""
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


print("\n--- DPO Data Construction Example ---")
dpo_pair = create_dpo_pair(
    prompt="Explain deep learning",
    good_response="Deep learning is a subset of machine learning using multi-layer neural networks to automatically learn hierarchical data representations. It has achieved breakthrough results in image recognition and NLP.",
    bad_response="Deep learning is just learning very deeply."
)
print(f"Chosen: {dpo_pair['chosen'][1]['content'][:50]}...")
print(f"Rejected: {dpo_pair['rejected'][1]['content']}")

# ============================================================================
# 6. Data Decontamination
# ============================================================================
"""
Decontamination: Ensure training data doesn't contain evaluation set content.

Why important?
  If training data includes evaluation questions and answers,
  the model "memorizes" them, inflating scores without real ability.

Methods:
  1. N-gram matching: Check if training data contains evaluation n-grams
  2. MinHash: Approximate matching for rewritten evaluation data
  3. Manual review: Human check of suspicious data
"""


def decontaminate(train_texts, eval_texts, n_gram_size=13):
    """Simple n-gram decontamination"""
    eval_ngrams = set()
    for text in eval_texts:
        tokens = text.split()
        for i in range(len(tokens) - n_gram_size + 1):
            ngram = ' '.join(tokens[i:i + n_gram_size])
            eval_ngrams.add(ngram)

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


# Demo decontamination
print("\n--- Data Decontamination Example ---")
eval_data = [
    "The capital of China is Beijing, this is common geographic knowledge",
]
train_data = [
    "Beijing is the capital of China, with a long history",  # Overlap
    "Machine learning is an important branch of artificial intelligence",  # No overlap
]

clean, n_contaminated = decontaminate(train_data, eval_data, n_gram_size=5)
print(f"Training data: {len(train_data)} items")
print(f"Contaminated: {n_contaminated} items")
print(f"After cleaning: {len(clean)} items")

# ============================================================================
# 7. Data Mixing Strategy
# ============================================================================
"""
Data mixing proportions significantly impact model performance.

Typical pretraining mix:
┌──────────────┬────────────┬──────────────────────────┐
│ Data Type    │ Proportion │ Description              │
├──────────────┼────────────┼──────────────────────────┤
│ Web text     │ 60-70%     │ General knowledge        │
│ Code         │ 10-15%     │ Reasoning ability        │
│ Books        │ 5-10%      │ Deep knowledge           │
│ Papers       │ 5%         │ Domain expertise         │
│ Encyclopedia │ 5%         │ Factual knowledge        │
│ Q&A          │ 5%         │ Conversational ability   │
└──────────────┴────────────┴──────────────────────────┘
"""

print("\n--- Data Mixing Strategy ---")
data_mix = {
    "Web text": 0.65,
    "Code": 0.12,
    "Books": 0.08,
    "Papers": 0.05,
    "Encyclopedia": 0.05,
    "Q&A": 0.05,
}
print("Pretraining data mix:")
for dtype, ratio in data_mix.items():
    bar = "█" * int(ratio * 40)
    print(f"  {dtype:12s}: {ratio:.0%} {bar}")

# ============================================================================
# 8. Tokenizer Training
# ============================================================================
"""
Tokenizer choice significantly impacts model performance:

1. BPE (Byte Pair Encoding)
   - Used by GPT series
   - Starts from characters, merges frequent byte pairs
   - Good for English

2. SentencePiece / BPE
   - Used by LLaMA series
   - Multilingual support
   - Chinese typically uses 32K-128K vocab

3. WordPiece
   - Used by BERT
   - Similar to BPE but different selection method

Vocab size impact:
  - Too small (<10K): Low encoding efficiency, sequences too long
  - Too large (>200K): Embedding matrix too large, parameter waste
  - Chinese recommended: 32K-64K
  - Multilingual recommended: 128K-256K
"""

print("\n--- Tokenizer Vocab Size Impact ---")
for vocab_size in [10000, 32000, 64000, 128000, 256000]:
    dim = 512
    emb_params = vocab_size * dim / 1e6
    print(f"  Vocab={vocab_size:>6d}, dim={dim} → Embedding params={emb_params:.1f}M")

# ============================================================================
# Exercises
# ============================================================================
print("\n" + "=" * 70)
print("Exercises")
print("=" * 70)

# Exercise 1: Data cleaning
print("\nExercise 1: Data cleaning")
dirty_text = "<p>This is HTML text&nbsp;&nbsp;with extra spaces and tags</p>"
cleaned = clean_text(dirty_text)
print(f"Original: {dirty_text}")
print(f"Cleaned: {cleaned}")

# Exercise 2: Quality filtering
print("\nExercise 2: Quality filtering")
test_texts = [
    "This is normal text with moderate length and meaningful content.",
    "Short",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
for text in test_texts:
    passed = filter_quality(text, min_length=10)
    print(f"  {'KEEP' if passed else 'FILTER'}: {text[:30]}...")

# Exercise 3: Exact deduplication
print("\nExercise 3: Exact deduplication")
dup_texts = ["Hello World", "Hello World", "Hi World", "Hello World", "Hi World"]
unique = exact_dedup(dup_texts)
print(f"Original: {len(dup_texts)} items")
print(f"After dedup: {len(unique)} items → {unique}")

# Exercise 4: SFT data validation
print("\nExercise 4: SFT data validation")
test_sample = {
    "conversations": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a widely-used high-level programming language."}
    ]
}
valid, reason = validate_sft_data(test_sample)
print(f"Result: {'VALID' if valid else 'INVALID'} - {reason}")

print("\n" + "=" * 70)
print("Lesson 25 Summary:")
print("  1. Data quality determines model quality: Garbage in, garbage out")
print("  2. Data cleaning: Remove HTML, normalize, quality filter")
print("  3. Data deduplication: Exact dedup + MinHash approximate dedup")
print("  4. Data decontamination: Ensure training data doesn't contain eval set content")
print("  5. Data mixing: Proportions of different data types affect model capabilities")
print("  6. Tokenizer: Vocab size affects encoding efficiency and parameter count")
print("=" * 70)
