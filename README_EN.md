# LLM-Zero: Learn LLM from Scratch

[![GitHub stars](https://img.shields.io/github/stars/1209244478/llm-zero?style=social)](https://github.com/1209244478/llm-zero)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![中文](https://img.shields.io/badge/README-中文-red.svg)](README.md)

> **26 lessons to master LLMs from scratch** — From Python basics to SFT/DPO/RLHF, each lesson is a standalone `.py` file you can run immediately!

[中文版](README.md)

Learn Large Language Models step by step, from zero to deployment.
No prerequisites required — from Python/PyTorch basics to full LLM deployment.

Based on the [MiniMind](https://github.com/jingyaogong/minimind) project, this learning repo breaks down every component of a small LLM into self-contained, runnable lessons.

## Project Highlights

- **Truly beginner-friendly** — Lesson 0 starts with "what is deep learning?", explains terminal, pip, import, and walks you through training your first model
- **Each lesson runs independently** — No GPU needed, no data to download, just `python lesson00.py` and go
- **Explains like you're five** — Every concept has intuitive analogy + math derivation + code implementation + running results, four layers of understanding
- **Bilingual** — `lessons/` in Chinese + `lessons_en/` in English, fully synchronized content
- **Complete coverage** — From Tokenizer to Multimodal, from pre-training to RLHF, 26 lessons cover the full LLM pipeline
- **Maps to MiniMind** — Each lesson maps to the corresponding MiniMind source file, so you can read the real code after learning
- **Rich intuitive analogies** — "Teaching a child to recognize cats" for deep learning, "clock" for RoPE frequency groups, "home renovation" for LoRA, "library" for FFN
- **Concrete numbers** — GQA saves 87.5% KV Cache, LoRA needs only 0.1% parameters, MoE router collapse before/after comparison
- **End-to-end training demos** — Lesson 0 trains your first model, Lesson 9 trains TinyGPT, Lesson 20 trains a complete MiniGPT

## Core Curriculum (Lessons 0–10) — Required, in order

| Lesson | Topic | Description |
|--------|-------|-------------|
| 0 | Basics | Python/PyTorch zero-to-hero intro |
| 1 | Tokenizer | How does text become numbers? (Char / Word / BPE) |
| 2 | Embedding | How do numbers become vectors? (Token + Position) |
| 3 | RMSNorm | Why do we need normalization? (Root Mean Square) |
| 4 | RoPE | How does the model know word positions? (Rotary Position Embedding) |
| 5 | Attention | The core mechanism: how words attend to each other (Single-Head) |
| 6 | Multi-Head Attention | MHA / GQA / MQA — parallel attention heads |
| 7 | FFN | Feed-Forward Network: refining memories after attention (SwiGLU) |
| 8 | Block | Assembling Attention + FFN into a Transformer layer |
| 9 | GPT | Stacking Blocks into a complete language model |
| 10 | Training | Training loop: loss, optimizer, scheduler, gradient clipping |
| 11 | Generation | Generation strategies: greedy, temperature, top-k/p, KV Cache |

## Advanced Curriculum (Lessons 12–19) — Advanced features & core knowledge

| Lesson | Topic | Description |
|--------|-------|-------------|
| 12 | Optimizers | SGD / Momentum / Adam / AdamW / Muon |
| 13 | Attention Variants | Linear Attention / ALiBi / Flash Attention |
| 14 | Mamba | State Space Models (SSM / Mamba) |
| 15 | LoRA | Low-Rank Adaptation fine-tuning |
| 16 | YaRN | Length extrapolation (train short, infer long) |
| 17 | mHC | Manifold-Constrained Hyper-Connections for deep networks |
| 18 | Quantization | Quantization & deployment (INT4/INT8/GPTQ/AWQ) |
| 19 | Speculative Decoding | Draft-then-verify for faster generation |

## Capstone (Lesson 20) — Putting it all together

| Lesson | Topic | Description |
|--------|-------|-------------|
| 20 | Final Project | Build a runnable, trainable mini LLM from scratch |

## Alignment & Optimization (Lessons 21–25) — Making models better

| Lesson | Topic | Description |
|--------|-------|-------------|
| 21 | SFT | Supervised Fine-Tuning: teaching the model to converse |
| 22 | DPO | Direct Preference Optimization: preferring good responses |
| 23 | Distillation | Knowledge Distillation: big model teaches small model |
| 24 | RLHF | RLHF / GRPO: aligning with human feedback |
| 25 | Data Prep | Data preparation & cleaning: garbage in, garbage out |
| 26 | Multimodal | Vision + Language: teaching the model to see (ViT + Projector + LLM) |

## Recommended Learning Paths

- **Complete beginner**: 0 → 1 → 2 → ... → 10 → 11 → 12 → ... → 19 → 20
- **Knows PyTorch**: 5 → 6 → ... → 10 → 11 → 12 → ... → 19 → 20
- **Algorithm focus**: 12 → 13 → 14 → 16 → 17
- **Deployment focus**: 15 → 18 → 19
- **Model alignment**: 21 → 22 → 24 → 23 → 25
- **Quick overview**: 1 → 5 → 9 → 11 → 20

## How to Run

Each lesson is a standalone Python file — just run it directly:

**Chinese version:**
```bash
python lessons/lesson00_basics.py
python lessons/lesson01_tokenizer.py
# ...
python lessons/lesson20_final_project.py   # Capstone
python lessons/lesson21_sft.py             # Supervised Fine-Tuning
python lessons/lesson22_dpo.py             # Direct Preference Optimization
python lessons/lesson23_distillation.py    # Knowledge Distillation
python lessons/lesson24_rlhf.py            # RLHF/GRPO
python lessons/lesson25_data_prep.py       # Data Preparation
python lessons/lesson26_multimodal.py      # Multimodal
```

**English version:**
```bash
python lessons_en/lesson00_basics.py
python lessons_en/lesson01_tokenizer.py
# ...
python lessons_en/lesson20_final_project.py   # Capstone
python lessons_en/lesson21_sft.py             # Supervised Fine-Tuning
python lessons_en/lesson22_dpo.py             # Direct Preference Optimization
python lessons_en/lesson23_distillation.py    # Knowledge Distillation
python lessons_en/lesson24_rlhf.py            # RLHF/GRPO
python lessons_en/lesson25_data_prep.py       # Data Preparation
python lessons_en/lesson26_multimodal.py      # Multimodal
```

## Dependencies

```bash
pip install torch
pip install numpy matplotlib  # optional, for plotting
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

## Lesson ↔ Original MiniMind Project Mapping

| Lesson | Original Project File/Component |
|--------|-------------------------------|
| 1 Tokenizer | model/ tokenizer config (vocab_size=6400, BPE) |
| 2 Embedding | model_minimind.py: `embed_tokens` |
| 3 RMSNorm | model_minimind.py: `RMSNorm` (eps=1e-5) |
| 4 RoPE | model_minimind.py: `precompute_freqs_cis` / `apply_rotary_pos_emb` |
| 5 Attention | model_minimind.py: `Attention` (GQA + QK-Norm) |
| 6 Multi-Head Attn | model_minimind.py: `Attention` (n_heads, n_kv_heads) |
| 7 FFN | model_minimind.py: `FeedForward` (SwiGLU) |
| 8 Block | model_minimind.py: `MiniMindBlock` (Pre-Norm + residual) |
| 9 GPT | model_minimind.py: `MiniMindModel` / `MiniMindForCausalLM` |
| 10 Training | 3-train_pt.py / 4-train_lora_pt.py |
| 11 Generation | model_minimind.py: `MiniMindForCausalLM.generate` (KV Cache) |
| 12 Optimizers | 3-train_pt.py: AdamW + Muon mixed optimizer |
| 13 Attn Variants | model_minimind.py: `LinearAttention` / `ALiBiAttention` |
| 14 Mamba | model_minimind.py: `MambaLayer` |
| 15 LoRA | model_minimind.py: `LoRAFeedForward` |
| 16 YaRN | model_minimind.py: `rope_scaling` config |
| 17 mHC | model_advanced.py: `ManifoldConstrainedHyperConnection` |
| 18 Quantization | Quantization & deployment configs |
| 19 Speculative | Draft-then-verify generation |
| 20 Final Project | Full pipeline integration |
| 21 SFT | trainer/train_full_sft.py (SFT dataset + Loss Mask) |
| 22 DPO | trainer/train_dpo.py (DPO loss + reference model) |
| 23 Distillation | trainer/train_distillation.py (KL divergence + temperature) |
| 24 RLHF | trainer/train_grpo.py (reward model + PPO/GRPO) |
| 25 Data Prep | dataset/lm_dataset.py (data cleaning + deduplication) |
| 26 Multimodal | Multimodal: ViT + Projector + LLM fusion |

## Project Structure

```
minimind-wrz-learn/
├── lessons/                    # Chinese lessons (中文课程)
│   ├── lesson00_basics.py
│   ├── lesson01_tokenizer.py
│   ├── ...
│   ├── lesson20_final_project.py
│   ├── lesson21_sft.py
│   ├── lesson22_dpo.py
│   ├── lesson23_distillation.py
│   ├── lesson24_rlhf.py
│   ├── lesson25_data_prep.py
│   └── lesson26_multimodal.py
├── lessons_en/                 # English lessons
│   ├── lesson00_basics.py
│   ├── lesson01_tokenizer.py
│   ├── ...
│   ├── lesson20_final_project.py
│   ├── lesson21_sft.py
│   ├── lesson22_dpo.py
│   ├── lesson23_distillation.py
│   ├── lesson24_rlhf.py
│   ├── lesson25_data_prep.py
│   └── lesson26_multimodal.py
├── README.md                   # Chinese README
├── README_EN.md                # English README (this file)
└── requirements.txt
```

## Key Concepts Covered

- **Tokenization**: Character-level, Word-level, BPE
- **Embedding**: Token embeddings, position encoding, weight sharing
- **Normalization**: RMSNorm vs LayerNorm
- **Position Encoding**: RoPE (Rotary Position Embedding)
- **Attention**: Single-head, Multi-head (MHA), Grouped Query (GQA), Multi-Query (MQA)
- **FFN**: SwiGLU with gate, up, down projections
- **Transformer Block**: Pre-norm + residual connections
- **GPT Architecture**: Embedding → Blocks → Norm → LM Head
- **Training**: Cross-entropy loss, AdamW, cosine schedule, gradient clipping
- **Generation**: Greedy, temperature, top-k, top-p, KV Cache
- **Optimizers**: SGD, Momentum, Adam, AdamW, Muon
- **Attention Variants**: Linear Attention, ALiBi, Flash Attention
- **Mamba**: State Space Models with selective mechanism
- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **YaRN**: Length extrapolation via frequency scaling
- **mHC**: Manifold-constrained hyper-connections for deep networks
- **Quantization**: INT8/INT4, GPTQ, AWQ, weight-only quantization
- **Speculative Decoding**: Draft-then-verify with acceptance/rejection
- **SFT**: Supervised Fine-Tuning with chat templates and loss masking
- **DPO**: Direct Preference Optimization with reference models
- **Knowledge Distillation**: Teacher-student training with KL divergence and temperature
- **RLHF/GRPO**: Reinforcement learning from human feedback, PPO, Group Relative Policy Optimization
- **Data Preparation**: Cleaning, deduplication, decontamination, and data mixing strategies
- **Multimodal**: Vision Encoder (ViT), Projector, vision-language fusion, image captioning

## Acknowledgments

This project is a learning companion to [MiniMind](https://github.com/jingyaogong/minimind) by Jingyao Gong. All core architecture and training logic reference the original MiniMind implementation.
