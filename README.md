# LLM-Zero: 从0学习大语言模型

[![GitHub stars](https://img.shields.io/github/stars/1209244478/llm-zero?style=social)](https://github.com/1209244478/llm-zero)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![English](https://img.shields.io/badge/README-English-blue.svg)](README_EN.md)

> **26课从零学会大语言模型** — 从 Python 基础到 SFT/DPO/RLHF，每课一个独立可运行的 `.py` 文件，0基础也能跟学！

从零开始，一步步理解大语言模型的每一个组件。
0基础也能学会！从 Python/PyTorch 基础到大模型部署，一课不缺。

## 基础课程（0-10）— 必学，按顺序

| 课程 | 主题 | 内容 |
|------|------|------|
| 第0课 | Basics | Python/PyTorch 0基础入门 |
| 第1课 | Tokenizer | 文本如何变成数字？ |
| 第2课 | Embedding | 数字如何变成向量？ |
| 第3课 | RMSNorm | 为什么需要归一化？ |
| 第4课 | RoPE | 模型如何知道词的位置？ |
| 第5课 | Attention | 核心机制：词与词如何相互关注？ |
| 第6课 | FFN | 前馈网络：注意力之后的记忆提炼 |
| 第7课 | Block | 把 Attention + FFN 组装成一个 Transformer 层 |
| 第8课 | GPT | 把多个 Block 组装成完整的语言模型 |
| 第9课 | Training | 训练循环：让模型学会说话 |
| 第10课 | Generation | 生成：让模型一个字一个字地写 |

## 进阶课程（11-19）— 高级特性与核心知识

| 课程 | 主题 | 内容 |
|------|------|------|
| 第11课 | Advanced | MoE/TTT/MTP/KV Cache/MSA 五大高级特性 |
| 第12课 | Optimizers | SGD/Momentum/Adam/AdamW/Muon 优化器 |
| 第13课 | AttnVar | Linear Attention/ALiBi/Flash 注意力变体 |
| 第14课 | Mamba | 状态空间模型 (SSM/Mamba) |
| 第15课 | LoRA | 低秩适配微调 (轻量化微调) |
| 第16课 | YaRN | 长度外推 (训练短, 推长) |
| 第17课 | mHC | 超连接 (Manifold-Constrained Hyper-Connections) |
| 第18课 | Quantize | 量化与部署 (INT4/INT8/AWQ/GPTQ) |
| 第19课 | SpecDec | 投机解码加速生成 (Speculative Decoding) |

## 综合实战（第20课）— 全部知识串联

| 课程 | 主题 | 内容 |
|------|------|------|
| 第20课 | FinalProj | 从零搭建一个能跑能学的迷你 LLM |

## 对齐与优化课程（21-25）— 让模型变得更好

| 课程 | 主题 | 内容 |
|------|------|------|
| 第21课 | SFT | 监督微调：让模型学会对话 |
| 第22课 | DPO | 直接偏好优化：让模型偏好好回答 |
| 第23课 | Distillation | 知识蒸馏：大模型教小模型 |
| 第24课 | RLHF | 强化学习人类反馈 / GRPO |
| 第25课 | DataPrep | 数据准备与清洗：Garbage in, garbage out |
| 第26课 | Multimodal | 多模态：让模型看图说话 (ViT + Projector + LLM) |

## 学习路线建议

- **完全0基础**: 0 → 1 → 2 → ... → 10 → 11 → 12 → ... → 19 → 20
- **有PyTorch基础**: 5 → 6 → ... → 10 → 11 → 12 → ... → 19 → 20
- **研究算法**: 11 → 12 → 13 → 14 → 17
- **部署工程**: 15 → 18 → 19
- **模型对齐**: 21 → 22 → 24 → 23 → 25

## 运行方式

每课都是独立的 Python 文件，直接运行即可：

```bash
python lessons/lesson00_basics.py
python lessons/lesson01_tokenizer.py
# ...
python lessons/lesson20_final_project.py   # 综合实战
python lessons/lesson21_sft.py             # 监督微调
python lessons/lesson22_dpo.py             # 直接偏好优化
python lessons/lesson23_distillation.py    # 知识蒸馏
python lessons/lesson24_rlhf.py            # RLHF/GRPO
python lessons/lesson25_data_prep.py       # 数据准备与清洗
python lessons/lesson26_multimodal.py      # 多模态
```

## 依赖

```bash
pip install torch
pip install numpy matplotlib  # 可选，用于绘图
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 课程与 MiniMind 原始项目对应关系

| 课程 | 原始项目文件/功能 |
|------|------------------|
| 第1课 Tokenizer | model/ tokenizer 配置 (vocab_size=6400, BPE) |
| 第2课 Embedding | model_minimind.py: `embed_tokens` |
| 第3课 RMSNorm | model_minimind.py: `RMSNorm` (eps=1e-5) |
| 第4课 RoPE | model_minimind.py: `precompute_freqs_cis` / `apply_rotary_pos_emb` |
| 第5课 Attention | model_minimind.py: `Attention` (GQA + QK-Norm) |
| 第6课 FFN | model_minimind.py: `FeedForward` (SwiGLU) |
| 第7课 Block | model_minimind.py: `MiniMindBlock` (Pre-Norm + 残差) |
| 第8课 GPT | model_minimind.py: `MiniMindModel` / `MiniMindForCausalLM` |
| 第9课 Training | 3-train_pt.py / 4-train_lora_pt.py |
| 第10课 Generation | model_minimind.py: `MiniMindForCausalLM.generate` (KV Cache) |
| 第11课 MoE | model_minimind.py: `MOEFeedForward` (aux_loss + norm_topk_prob) |
| 第11课 TTT | model_minimind.py: `TTTFeedForward` |
| 第11课 MTP | model_minimind.py: `MTPHead` |
| 第11课 KV Cache | model_minimind.py: `KVCache` |
| 第13课 AttnVar | model_minimind.py: `LinearAttention` / `ALiBiAttention` |
| 第14课 Mamba | model_minimind.py: `MambaLayer` |
| 第15课 LoRA | model_minimind.py: `LoRAFeedForward` |
| 第16课 YaRN | model_minimind.py: `rope_scaling` 配置 |
| 第18课 Quantize | 量化部署相关配置 |
| 第20课 FinalProj | 完整流程串联 |
| 第21课 SFT | trainer/train_full_sft.py (SFT数据集 + Loss Mask) |
| 第22课 DPO | trainer/train_dpo.py (DPO损失 + 参考模型) |
| 第23课 Distillation | trainer/train_distillation.py (KL散度 + 温度) |
| 第24课 RLHF | trainer/train_grpo.py (奖励模型 + PPO/GRPO) |
| 第25课 DataPrep | dataset/lm_dataset.py (数据清洗 + 去重) |
| 第26课 Multimodal | 多模态：ViT + Projector + LLM 融合 |
