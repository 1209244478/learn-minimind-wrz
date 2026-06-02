"""
Lesson 1: Tokenizer — How Does Text Become Numbers?
====================================================

Why do models need a Tokenizer?
  Computers only understand numbers, not text.
  All AI models (LLM, CV, Speech) need to convert inputs to numbers.

What does a Tokenizer do?
  Text "Hello World"  ->  Number sequence [1234, 5678, 9012, 3456]

Three mainstream tokenization approaches:

  1. Character-level
     Principle: Each character is one token
     Example: "Hello" -> ['H', 'e', 'l', 'l', 'o'] -> [45, 67, 12, 12, 89]
     Pros: Small vocabulary (~100 tokens)
     Cons: Long sequences, slow training

  2. Word-level
     Principle: Each word is one token
     Example: "Hello World" -> ['Hello', 'World'] -> [1234, 5678]
     Pros: Clear semantics
     Cons: Vocabulary explosion (hundreds of thousands), OOV problem

  3. Subword-level (BPE)
     Principle: Merge frequent word pairs, split rare words
     Example: "unhappiness" -> ['un', 'happiness'] -> [123, 456]
     Pros: Balances vocabulary size and sequence length
     Cons: Complex implementation

Analogy:
  A Tokenizer is like a "codebook"
  - Chinese: characters -> numbers
  - English: words -> numbers
  - The model can only see numbers, not the original text

Run: python lessons_en/lesson01_tokenizer.py
"""

import torch
import re
from collections import Counter, defaultdict


# ============================================================
# Part 1: Character-Level Tokenizer (Simplest)
# ============================================================
print("=" * 60)
print("Part 1: Character-Level Tokenizer")
print("=" * 60)


class CharTokenizer:
    """Character-level tokenizer - each character is one token"""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return ''.join(self.id_to_char[i] for i in ids)


# Experiment
sample = "hello world"
tokenizer = CharTokenizer(sample)
ids = tokenizer.encode(sample)
print(f"\nOriginal: '{sample}'")
print(f"Encoded:  {ids}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Sequence length: {len(ids)}")
print(f"Decoded: '{tokenizer.decode(ids)}'")

# Pros and cons
print("\nCharacter-Level Tokenizer Pros/Cons:")
print("  + Small vocabulary (dozens of characters)")
print("  - Long sequences (1 token per character)")
print(f"  Example: 100-char text -> {100} tokens, 10000-char text -> {10000} tokens")


# ============================================================
# Part 2: Word-Level Tokenizer
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Word-Level Tokenizer")
print("=" * 60)


class WordTokenizer:
    """Word-level tokenizer - each word is one token"""

    def __init__(self, text):
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


# Training text
corpus = """
The quick brown fox jumps over the lazy dog.
The dog barks at the fox.
The fox runs away quickly.
Machine learning is a branch of artificial intelligence.
Deep learning is a subset of machine learning.
Neural networks are the core of deep learning.
"""

word_tokenizer = WordTokenizer(corpus)
print(f"\nVocab size: {word_tokenizer.vocab_size}")
print(f"First 20 words: {word_tokenizer.words[:20]}")

# Test
test = "The fox is happy"
ids = word_tokenizer.encode(test)
print(f"\nOriginal: '{test}'")
print(f"Tokenized: {word_tokenizer._tokenize(test)}")
print(f"Encoded: {ids}")
print(f"Decoded: '{word_tokenizer.decode(ids)}'")

print("\nWord-Level Tokenizer Pros/Cons:")
print("  + Clear semantics")
print("  - Large vocabulary (hundreds of thousands)")
print("  - OOV problem (unseen words -> UNK)")
print("  - Morphological variants: run/runs/running each count as a separate word")


# ============================================================
# Part 3: BPE (Byte Pair Encoding) Algorithm
# ============================================================
print("\n" + "=" * 60)
print("Part 3: BPE Algorithm")
print("=" * 60)

print("""
BPE is the method used by GPT, Llama, MiniMind, and all mainstream LLMs.
Core idea: Start from characters, keep merging the most frequent adjacent pairs

Algorithm flow:
  1. Initialize vocabulary = all single characters
  2. Count the frequency of adjacent character pairs
  3. Find the most frequent pair (a, b), merge into new token "ab"
  4. Add "ab" to the vocabulary
  5. Repeat 2-4 until vocabulary reaches target size
""")


def get_pairs(word):
    """Get all adjacent character pairs in a word"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def bpe_train(corpus, num_merges=10, end_of_word='</w>'):
    """BPE training algorithm"""
    # 1. Initialize: split each word into characters
    word_freqs = Counter(corpus.split())
    vocab = set()
    for word in word_freqs:
        vocab.update(list(word))
    vocab.add(end_of_word)

    # Each word represented as a list of characters
    splits = {word: [c for c in word] + [end_of_word] for word in word_freqs}

    print(f"Initial vocab: {sorted(vocab)[:20]}... ({len(vocab)} tokens)")

    # 2-N. Iterative merging
    merges = []
    for i in range(num_merges):
        # Count adjacent pair frequencies
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            splits_word = splits[word]
            for pair in get_pairs(splits_word):
                pair_freqs[pair] += freq

        if not pair_freqs:
            break

        # Find the most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]

        # Merge
        new_token = ''.join(best_pair)
        merges.append((best_pair, new_token))
        vocab.add(new_token)

        # Update all word splits
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

        print(f"  Merge #{i+1}: {best_pair} -> '{new_token}' (freq: {best_freq})")

    return vocab, merges, splits


# Prepare corpus
bpe_corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider"
print(f"\nTraining corpus: '{bpe_corpus}'")

vocab, merges, splits = bpe_train(bpe_corpus, num_merges=8)

print(f"\nFinal vocab size: {len(vocab)}")
print(f"Learned merge rules: {merges}")


# ============================================================
# Part 4: BPE Encoding/Decoding
# ============================================================
print("\n" + "=" * 60)
print("Part 4: BPE Encoding/Decoding")
print("=" * 60)


class SimpleBPE:
    """Simplified BPE tokenizer"""

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {t: i for i, t in enumerate(sorted(vocab))}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def _tokenize_word(self, word):
        """Apply BPE to a single word"""
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
print(f"\nOriginal: '{test}'")
print(f"Encoded:  {ids}")
print(f"Tokens:   {[bpe.id_to_token[i] for i in ids]}")
print(f"Decoded: '{bpe.decode(ids)}'")


# ============================================================
# Part 5: Full Comparison
# ============================================================
print("\n" + "=" * 60)
print("Part 5: Three Tokenizer Approaches Compared")
print("=" * 60)

test_text = "The quick brown fox jumps over the lazy dog"
print(f"\nTest text: '{test_text}' ({len(test_text)} characters)")

# Character-level
char_ids = CharTokenizer(test_text).encode(test_text)
print(f"\nCharacter-level: {len(char_ids)} tokens")
print(f"  First 10: {char_ids[:10]}")

# Word-level
word_ids = WordTokenizer(corpus + " " + test_text).encode(test_text)
print(f"\nWord-level: {len(word_ids)} tokens")
print(f"  Encoded: {word_ids}")

# Summary
print("\nComparison:")
print("+--------------+----------+----------+----------+")
print("| Approach     | Vocab    | Seq Len  | OOV      |")
print("+--------------+----------+----------+----------+")
print("| Char-level   | ~100     | Long     | None     |")
print("| Word-level   | 100K+    | Short    | UNK      |")
print("| BPE          | 10-100K  | Medium   | Fallback |")
print("+--------------+----------+----------+----------+")


# ============================================================
# Part 6: MiniMind's Tokenizer
# ============================================================
print("\n" + "=" * 60)
print("Part 6: MiniMind's Actual Tokenizer")
print("=" * 60)

print("""
MiniMind uses a pre-trained BPE tokenizer with vocabulary size ~6400

Usage:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("minimind")

# Encode
text = "The weather is nice today"
ids = tokenizer.encode(text)
print(ids)        # [101, 234, 567, 890, 102]

# Decode
text2 = tokenizer.decode(ids)
print(text2)      # "The weather is nice today"

# Batch processing
batch = tokenizer(["Hello", "Goodbye"], padding=True, return_tensors="pt")
```

MiniMind's characteristics:
  - Small vocabulary (~6400), fast training
  - Chinese + English + digits + punctuation
  - Uses BPE, balancing vocabulary and efficiency
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Character Statistics
  Given the text "Hello, World! 123"
  How many unique characters are there? What is the sequence length after CharTokenizer encoding?

[Exercise 2] Word-Level OOV Problem
  Vocabulary only contains ["I", "love", "coding"]
  What happens when encoding "I love music"?
  What problem does this illustrate?

[Exercise 3] BPE Merge Count
  If we want a vocabulary of size 100 and start with 50 initial characters,
  how many merges are needed?

[Exercise 4] BPE Advantage
  Suppose there are 1 million different English words.
  With BPE vocabulary size 32000, how much does the vocabulary shrink
  compared to pure word-level? How much does sequence length increase?

[Exercise 5] Implement a Simplified BPE
  Input: "a a a a b b c"
  Goal: 3 merges, write out the pair for each merge
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

# Exercise 1
print("\n[Exercise 1 Answer]")
text1 = "Hello, World! 123"
ct = CharTokenizer(text1)
ids1 = ct.encode(text1)
print(f"  Text: '{text1}'")
print(f"  Unique characters: {ct.vocab_size}")
print(f"  Encoded sequence length: {len(ids1)}")
print(f"  Encoded: {ids1}")

# Exercise 2
print("\n[Exercise 2 Answer]")
print("  Encoding 'I love music': ['I', 'love', 'music']")
print("  'music' is not in the vocabulary -> encoded as [-1] or <UNK>")
print("  This illustrates the OOV (Out-Of-Vocabulary) problem")

# Exercise 3
print("\n[Exercise 3 Answer]")
print("  Target vocab: 100")
print("  Initial: 50")
print("  Each merge adds +1 token")
print("  Merges needed: 100 - 50 = 50")

# Exercise 4
print("\n[Exercise 4 Answer]")
print("  Pure word-level: 1M words")
print("  BPE: 32K words (96.8% reduction)")
print("  Trade-off: each word is split into 1.5-2 subwords on average")
print("  Sequence length increases 50-100%, but vocabulary shrinks 30x")

# Exercise 5
print("\n[Exercise 5 Answer]")
text5 = "a a a a b b c"
print(f"  Text: '{text5}'")
print(f"  Initial: ['a', 'a', 'a', 'a', 'b', 'b', 'c']")
print(f"  Adjacent pair frequencies: ('a','a')=3, ('a','b')=1, ('b','b')=1, ('b','c')=1")
print()
print(f"  Merge 1: ('a', 'a') freq 3 -> 'aa'")
print(f"    Sequence becomes: ['aa', 'a', 'a', 'b', 'b', 'c']")
print(f"    Adjacent pair frequencies: ('aa','a')=1, ('a','a')=1, ('a','b')=1, ('b','b')=1, ('b','c')=1")
print()
print(f"  Merge 2: ('b', 'b') freq 1 -> 'bb'  (multiple pairs with freq=1, pick any)")
print(f"    Sequence becomes: ['aa', 'a', 'a', 'bb', 'c']")
print(f"    Adjacent pair frequencies: ('aa','a')=1, ('a','a')=1, ('a','bb')=1, ('bb','c')=1")
print()
print(f"  Merge 3: ('aa', 'a') freq 1 -> 'aaa'  (pick any)")
print(f"    Sequence becomes: ['aaa', 'a', 'bb', 'c']")
print(f"    Final vocabulary: a, b, c, aa, bb, aaa")


# ============================================================
# Lesson Summary
# ============================================================
print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Tokenizer's Role
   - Converts text to numbers so the model can process it
   - Trade-off between vocabulary size and sequence length

2. Three Approaches
   - Character-level: small vocab, long sequences
   - Word-level: clear semantics, vocabulary explosion
   - BPE: balanced approach, the choice of all modern LLMs

3. BPE Principle
   - Start from characters, keep merging frequent adjacent pairs
   - Balances vocabulary size and sequence length
   - No OOV problem (can fall back to characters)

4. MiniMind Uses BPE
   - Vocabulary size ~6400
   - Fast training and inference
   - Supports Chinese and English

5. Practical Usage
   - Use the transformers library's AutoTokenizer
   - encode/decode in one line of code

Next: Lesson 2 - Embedding (How do numbers become vectors?)
""")
