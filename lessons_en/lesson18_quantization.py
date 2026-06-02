"""
Lesson 18: Quantization and Deployment — Running Large Models on Consumer Devices
==================================================================================

Problem: A 7B model's fp32 weights are 28GB — how to run it on a 16GB consumer GPU?

Quantization:
  Convert fp32 (4 bytes) parameters to lower precision (2 bytes / 1 byte)
  - fp32 -> fp16: 2x compression
  - fp32 -> int8: 4x compression
  - fp32 -> int4: 8x compression
  - Smaller model size, faster inference, slight accuracy loss

Common approaches:
  - PTQ (Post-Training Quantization): Directly quantize a trained model
  - QAT (Quantization-Aware Training): Account for quantization during training
  - GPTQ, AWQ, GGUF: Different algorithms

This lesson covers:
  1. Quantization basics (symmetric/asymmetric, per-tensor/per-channel)
  2. Core algorithms (MinMax, simplified GPTQ)
  3. Inference deployment (ONNX, TensorRT)

Run: python lessons_en/lesson18_quantization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: Why Quantization is Needed
# ============================================================

print("=" * 60)
print("Part 1: Why Quantization is Needed")
print("=" * 60)

print("""
The "memory wall" problem for models:
  - 7B model fp32: 28GB weights
  - Need at least 2x GPU memory (weights + activations) ≈ 56GB
  - Consumer GPU RTX 4090 has 24GB — can't fit it

Benefits of quantization:
  + Reduce 2-8x memory usage
  + Faster inference (integer ops are faster than floating point)
  + Lower power consumption (mobile-friendly)
  + Lower deployment barrier (CPU can also run)

Costs of quantization:
  - Accuracy loss (1-5% perplexity increase)
  - Cannot directly use during training (needs QAT)
  - Different layers may have different sensitivity
""")

print("\n[Demo] Storage size at different precisions")
print("-" * 60)
print(f"{'Precision':<12}{'Bits':<8}{'7B Model Weights':<18}{'Compression':<10}")
print("-" * 60)
for name, bits, factor in [("FP32", 32, 1), ("FP16/BF16", 16, 2), ("INT8", 8, 4), ("INT4", 4, 8)]:
    size_gb = 7 * 4 / factor
    print(f"{name:<12}{bits:<8}{size_gb:<18.1f}{factor}x")


# ============================================================
# Part 2: Quantization Basics
# ============================================================

print("\n" + "=" * 60)
print("Part 2: Quantization Basics")
print("=" * 60)

print("""
Core idea of quantization: Map continuous floating-point values to discrete integers

Two approaches:

1) Symmetric Quantization:
   Assumes data distribution is symmetric around 0
   x_int = round(x / scale)         scale = max(|x|) / (qmax - qmin)
   x_dequant = x_int * scale

   Pros: Simple, hardware-friendly
   Cons: Wastes quantization points when data is biased

2) Asymmetric Quantization:
   Considers the min/max values of the data
   scale = (max - min) / (qmax - qmin)
   zero_point = round(qmin - min / scale)
   x_int = round(x / scale) + zero_point

   Pros: Higher accuracy
   Cons: Slightly more complex

Granularity:

1) Per-tensor: One scale/zero_point for the entire tensor
2) Per-channel: One scale/zero_point per channel (more accurate)
3) Per-group: One scale per group of 32/64 elements (used by GPTQ)
""")


# ============================================================
# Part 3: Symmetric Quantization Implementation
# ============================================================

print("\n" + "=" * 60)
print("Part 3: Symmetric Quantization Implementation")
print("=" * 60)


def quantize_per_tensor_symmetric(tensor, n_bits=8):
    """Per-tensor symmetric quantization"""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -2 ** (n_bits - 1)

    abs_max = tensor.abs().max()
    scale = abs_max / qmax if abs_max > 0 else 1.0

    tensor_int = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)
    return tensor_int, scale.item()


def dequantize_per_tensor_symmetric(tensor_int, scale, dtype=torch.float32):
    """Per-tensor symmetric dequantization"""
    return tensor_int.to(dtype) * scale


print("\n[Demo] Symmetric quantization example")
print("-" * 60)
print("Original weights (FP32):")
W = torch.tensor([[-1.5, 0.2, 0.8], [0.3, -2.1, 1.7], [0.1, 0.5, -0.9]])
print(W)
print(f"  Bytes: {W.numel() * 4}")

W_int8, scale = quantize_per_tensor_symmetric(W, n_bits=8)
print(f"\nQuantized (INT8), scale={scale:.4f}:")
print(W_int8)
print(f"  Bytes: {W_int8.numel() * 1}")

W_dequant = dequantize_per_tensor_symmetric(W_int8, scale)
print(f"\nDequantized:")
print(W_dequant)

error = (W - W_dequant).abs()
print(f"\nMax absolute error: {error.max():.4f}")
print(f"Average relative error: {(error / (W.abs() + 1e-8)).mean():.2%}")


# ============================================================
# Part 4: Simplified GPTQ (Per-Group Quantization)
# ============================================================

print("\n" + "=" * 60)
print("Part 4: Simplified GPTQ (Per-Group Quantization)")
print("=" * 60)

print("""
GPTQ core idea:
  - Quantize per group (32/64/128 elements)
  - Use Hessian matrix to guide quantization, minimizing output error
  - Quantize column by column, adjust remaining columns after each to compensate

Simplified flow:
  1. Compute Hessian: H = X^T·X
  2. Initialize errors = 0
  3. for each column i:
     a. Quantize column i: w_q = quant(w)
     b. Compute quantization error: e = w - w_q
     c. Update subsequent columns: w[j] -= e * (H^-1)[i, j] / (H^-1)[i, i]
     d. Accumulate error
""")


def quantize_per_group(W, group_size=32, n_bits=4):
    """Per-group quantization (GPTQ style)"""
    out_features, in_features = W.shape
    qmax = 2 ** (n_bits - 1) - 1

    W_int = torch.zeros_like(W, dtype=torch.int8)
    scales = torch.zeros((out_features, in_features // group_size))

    for i in range(0, in_features, group_size):
        w_group = W[:, i:i+group_size]
        abs_max = w_group.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = abs_max / qmax
        w_int = torch.round(w_group / scale).clamp(-qmax-1, qmax).to(torch.int8)
        W_int[:, i:i+group_size] = w_int
        scales[:, i // group_size] = scale.squeeze()

    return W_int, scales


print("\n[Demo] Per-group quantization")
print("-" * 60)
W = torch.randn(4, 128) * 0.5

W_int, scales = quantize_per_group(W, group_size=32, n_bits=4)
print(f"Original: {W.shape} ({W.numel() * 4} bytes)")
print(f"Quantized: W_int {W_int.shape} ({W_int.numel() * 1} bytes) + scales {scales.shape} ({scales.numel() * 4} bytes)")
print(f"Compression ratio: {(W.numel() * 4) / (W_int.numel() * 1 + scales.numel() * 4):.1f}x")


# ============================================================
# Part 5: Weight-Only Quantization (WnA16)
# ============================================================

print("\n" + "=" * 60)
print("Part 5: Weight-Only Quantization")
print("=" * 60)

print("""
Weight-Only quantization (mainstream LLM deployment approach):
  - Quantize weights to INT4/INT8
  - Keep activations in FP16
  - At inference: Dynamically dequantize weights to FP16, compute normally

Why Weight-Only?
  - LLM inference is memory-bound, compute is not the bottleneck
  - Reducing weight reads accelerates inference
  - Activations stay in FP16, minimal accuracy loss

Performance comparison (RTX 4090, 7B model):
  +------------+----------+----------+------------+
  | Precision  | Memory   | Throughput| Accuracy Loss|
  +------------+----------+----------+------------+
  | FP16       | 14GB     | Baseline | 0%         |
  | INT8       | 7.5GB    | 1.5x     | <0.5%      |
  | INT4       | 4.2GB    | 2.5x     | 1-2%       |
  +------------+----------+----------+------------+
""")


# ============================================================
# Part 6: Quantization in MiniMind
# ============================================================

print("\n" + "=" * 60)
print("Part 6: Quantization in MiniMind")
print("=" * 60)

print("""
Using AutoGPTQ for INT4 quantization:

```bash
pip install auto-gptq
```

```python
from auto_gptq import BaseQuantizeConfig, AutoGPTQForCausalLM
from transformers import AutoTokenizer

model_path = "minimind"
quantized_model_path = "minimind-int4"

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
calibration_data = [...]
model.quantize(calibration_data)

model.save_quantized(quantized_model_path)
```

Using bitsandbytes (built into Transformers):

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "minimind",
    quantization_config=bnb_config,
    device_map="auto",
)
```
""")


# ============================================================
# Part 7: Deployment Tool Comparison
# ============================================================

print("\n" + "=" * 60)
print("Part 7: Deployment Tool Comparison")
print("=" * 60)

print("""
Mainstream deployment solutions:

1) llama.cpp / GGUF
   - Pure C++ implementation, extremely high performance
   - Supports CPU/GPU
   - Rich quantization formats (Q2_K, Q4_K_M, Q5_K_M, Q8_0)
   - Best for personal computer deployment

   Command: ./llama.cpp/main -m model.gguf -p "hello"

2) vLLM
   - PagedAttention memory management
   - 10-20x higher throughput
   - Best for server-side deployment

   ```python
   from vllm import LLM, SamplingParams
   llm = LLM(model="minimind-int4", quantization="gptq")
   ```

3) TensorRT-LLM
   - NVIDIA's official optimization
   - FP8/INT4 extreme optimization
   - Requires NVIDIA GPU

4) ONNX Runtime
   - Cross-platform
   - Integrate into various applications
   - Medium performance
""")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Lesson Summary")
print("=" * 60)

print("""
1. Quantization Basics
   - Floating point -> Integer conversion
   - Symmetric vs Asymmetric
   - Per-tensor vs Per-group

2. Main Algorithms
   - MinMax: Simplest, moderate performance
   - GPTQ: Hessian-guided, SOTA
   - AWQ: Activation-aware, good for LLMs
   - NF4: Used by bitsandbytes, 4-bit friendly

3. LLM Deployment
   - Weight-Only quantization is mainstream
   - INT4 + FP16 activations is the best balance
   - 3-4x memory savings, <2% accuracy loss

4. Deployment Tools
   - llama.cpp: Best for personal computers
   - vLLM: Best for server-side
   - TensorRT-LLM: Best for NVIDIA GPU

5. In MiniMind
   - AutoGPTQ: Most commonly used
   - bitsandbytes: Simplest
   - Quantize then deploy to llama.cpp

Next: Lesson 19 - Speculative Decoding for Faster Generation
""")


# ============================================================
# Deep Understanding: The Art of "Photo Compression" in Quantization
# ============================================================
print("\n" + "=" * 60)
print("Deep Understanding: The Art of 'Photo Compression' in Quantization")
print("=" * 60)

print("""
[Analogy: Quantization = Photo Compression]
-------------------------------------------

  Original photo (FP32):
    10000x10000 pixels, 32 bits per pixel
    -> Perfect, 400 MB

  Compressed photo (INT8):
    10000x10000 pixels, 8 bits per pixel
    -> Almost the same, 100 MB (4x compression)

  Highly compressed (INT4):
    4 bits per pixel
    -> Slightly blurry, 50 MB (8x compression)

  Extreme compression (INT2):
    2 bits per pixel
    -> Mosaic, 25 MB (16x compression)

  LLM quantization:
    Model parameters are like "pixels"
    Quantization precision is like "compression ratio"
    -> Store the model in less space


[Why Quantization?]
-------------------

  1. Memory pressure:
    LLaMA-70B (FP32): 280 GB -> Unusable
    LLaMA-70B (FP16): 140 GB -> Needs multi-GPU
    LLaMA-70B (INT4): 35 GB  -> Single GPU possible!

  2. Inference speed:
    Memory bandwidth is the bottleneck (not compute)
    Quantization reduces data movement
    -> INT4 is 2-4x faster than FP16

  3. Deployment cost:
    Cloud GPUs are expensive
    Quantized models use smaller GPUs
    -> 50%+ cost reduction

  4. Edge deployment:
    Mobile / embedded devices
    Limited memory, quantization is essential
    -> 4-bit models fit on phones


[Quantization Precision Comparison]
------------------------------------

  +--------+--------+----------+-------------+--------+
  | Prec.  | Bits   | 7B Mem   | Accuracy Loss| Speed  |
  +--------+--------+----------+-------------+--------+
  | FP32   | 32 bit | 28 GB    | 0%          | 1x     |
  | FP16   | 16 bit | 14 GB    | ~0%         | 1.5-2x |
  | BF16   | 16 bit | 14 GB    | ~0%         | 1.5-2x |
  | INT8   | 8 bit  | 7 GB     | <1%         | 1.5-2x |
  | INT4   | 4 bit  | 3.5 GB   | 2-5%        | 2-4x   |
  | INT3   | 3 bit  | 2.6 GB   | 5-10%       | 3-5x   |
  | INT2   | 2 bit  | 1.8 GB   | 20%+        | 5-10x  |
  +--------+--------+----------+-------------+--------+

  Sweet spot:
    INT4 / INT8 has almost no loss, best cost-effectiveness
    INT2 has large losses, use with caution


[Quantization Method Classification]
-------------------------------------

  +------------+------------+------------+------------+
  | Category   | Precision  | Speed      | Difficulty |
  +------------+------------+------------+------------+
  | PTQ        | 8/4 bit    | Very fast  | Simple     |
  | QAT        | 8/4 bit    | Slow       | Complex    |
  | GPTQ       | 4 bit      | Fast       | Medium     |
  | AWQ        | 4 bit      | Fast       | Medium     |
  | bits&bytes | 4/8 bit    | Medium     | Simple     |
  +------------+------------+------------+------------+

  PTQ (Post-Training Quantization):
    Directly quantize, no retraining
    -> Done in minutes
    -> 1-5% accuracy loss

  QAT (Quantization-Aware Training):
    Simulate quantization during training
    -> Requires full training
    -> < 1% accuracy loss

  GPTQ (most popular):
    Column-wise quantization with error compensation
    -> Takes tens of minutes
    -> INT4 with almost no loss

  AWQ:
    Protect important weights, quantize the rest
    -> Slightly slower than GPTQ
    -> Slightly better results


[Mathematical Principles of Quantization]
------------------------------------------

  Symmetric quantization:
    Map [-max, max] to [-127, 127] (INT8)
    scale = max / 127
    quantized = round(x / scale)

    Dequantization:
    dequantized = quantized * scale

  Asymmetric quantization:
    Map [min, max] to [0, 255] (UINT8)
    scale = (max - min) / 255
    zero_point = round(-min / scale)
    quantized = round(x / scale) + zero_point

    Dequantization:
    dequantized = (quantized - zero_point) * scale

  Example (INT8 symmetric):
    Weights: [0.1, 0.5, -0.3, 0.8]
    max = 0.8
    scale = 0.8 / 127 ≈ 0.0063
    Quantized: [16, 79, -48, 127]
    Dequantized: [0.101, 0.498, -0.302, 0.799]
    Error: < 0.01 (0.8%)


[Per-Channel vs Per-Tensor]
----------------------------

  Per-Tensor (coarse):
    One shared scale for the entire tensor
    -> Simple, but less accurate
    -> Outliers "flatten" other values

  Per-Channel (fine):
    Independent scale per channel
    -> More complex, but more accurate
    -> Industry default

  Per-Group (finest):
    One scale per 32/64/128 elements
    -> Used by GPTQ / AWQ
    -> Best accuracy


[Sources of Quantization Error]
---------------------------------

  1. Rounding error:
    True value 0.123 -> INT4 doesn't have 0.123 precision
    -> Quantized to 0.125, error 0.002
    -> Accumulates across most parameters

  2. Outliers:
    Most weights in [-0.1, 0.1]
    A few weights reach ±5
    -> max=5 determines scale
    -> Most values compressed near 0, accuracy lost

  Solutions:
    - Clip outliers (clamp)
    - Handle outlier channels separately (AWQ)
    - Group quantization (GPTQ)


[GPTQ's Core Idea]
-------------------

  Problem: Per-layer quantization error accumulates

  Solution: Use second-order information (Hessian matrix) to compensate

  Steps:
    1. Compute H = X^T X (input covariance)
    2. Quantize weights column by column
    3. After quantizing each column, adjust unquantized columns to compensate
    4. -> Error doesn't accumulate, high accuracy

  Complexity:
    One calibration pass (a few hundred samples)
    -> Done in tens of minutes

  Results:
    INT4 LLaMA-70B:
      Plain PTQ:  perplexity 8+
      GPTQ:       perplexity 5.5
      (Close to FP16's 5.4)


[AWQ's Core Idea]
------------------

  Observation:
    Not all weights are equally important
    0.1-1% of "salient weights" determine the output

  Strategy:
    1. Identify salient weights (large activation values)
    2. Don't quantize these, keep FP16
    3. Quantize other weights to INT4

  Advantage:
    Salient weights determine quality
    Quantizing other weights saves space
    -> Minimal accuracy loss

  Comparison:
    GPTQ: Modifies weight values
    AWQ:  Changes "which weights to quantize"


[Mixed-Precision Quantization]
-------------------------------

  Idea: Different layers use different precision

  Strategy:
    - Embedding layer: FP16 (important, large)
    - Attention layers: INT4 (repetitive structure)
    - FFN layers: INT4 (large parameter count)
    - LM Head: FP16 (output-critical)

  Effect:
    Overall precision INT4.5
    Key layers retain precision
    -> Performance close to INT8, memory close to INT4


[Quantization vs Distillation vs Pruning]
------------------------------------------

  +------------+------------+------------+------------+
  | Technique  | Principle  | Speed      | Accuracy   |
  +------------+------------+------------+------------+
  | Quantize   | Lower prec.| Very fast  | 95-99%     |
  | Distill    | Small learns| Slow (train)| 90-95%    |
  | Prune      | Remove red.| Fast       | 90-95%     |
  +------------+------------+------------+------------+

  Quantization advantages:
    - No retraining needed
    - Done in minutes
    - Best accuracy retention
    -> Industry's first choice


[Deployment Frameworks]
------------------------

  +--------------+------------+------------------+
  | Framework    | Strength   | Use Case         |
  +--------------+------------+------------------+
  | llama.cpp    | Cross CPU  | Edge/local       |
  | vLLM         | Server     | High throughput  |
  | TensorRT-LLM | NVIDIA opt | Production GPU   |
  | TGI          | HuggingFace| Cloud service    |
  | MLC-LLM      | Mobile/Web | Phone/browser    |
  | ONNX Runtime | Cross-plat | Multi-platform   |
  +--------------+------------+------------------+

  Recommendations:
    Local experience: llama.cpp (CPU/GPU both work)
    Server deployment: vLLM (highest throughput)
    Maximum performance: TensorRT-LLM
""")


# ============================================================
# Exercises
# ============================================================
print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

print("""
[Exercise 1] Quantization Benefits
  How much memory does LLaMA-7B need in FP16? In INT4?

[Exercise 2] Symmetric vs Asymmetric
  What scenarios are symmetric and asymmetric quantization best for?

[Exercise 3] PTQ vs QAT
  What are the pros and cons of Post-Training Quantization (PTQ)
  and Quantization-Aware Training (QAT)?

[Exercise 4] Per-Group Quantization
  Why is per-group quantization more accurate than per-tensor?

[Exercise 5] GPTQ Core
  What is GPTQ's key innovation? How does it reduce error?

[Exercise 6] Deployment Selection
  For each scenario, which deployment tool would you choose?
  a) Running on a personal laptop
  b) Serving 100 concurrent users
  c) Maximum throughput on NVIDIA GPU
""")


# ============================================================
# Exercise Answers
# ============================================================
print("\n" + "=" * 60)
print("Exercise Answers")
print("=" * 60)

print("\n[Exercise 1 Answer]")
print("  LLaMA-7B memory requirements:")
print()
print("  FP16: 7B × 2 bytes = 14 GB")
print("  INT4: 7B × 0.5 bytes = 3.5 GB")
print()
print("  With optimizer state (training):")
print("    FP32: 7B × 4 bytes × 3 (weights + grad + Adam) ≈ 84 GB")
print("    FP16: 7B × 2 bytes × 3 ≈ 42 GB")
print()
print("  Inference only:")
print("    FP16: ~14 GB (fits on RTX 4090)")
print("    INT4: ~3.5 GB (fits on almost any GPU)")

print("\n[Exercise 2 Answer]")
print("  Symmetric quantization:")
print("    Best for: Weights with symmetric distribution (common after normalization)")
print("    Advantages: Simple, hardware-friendly, no zero_point needed")
print("    Common use: Weight quantization in most LLM frameworks")
print()
print("  Asymmetric quantization:")
print("    Best for: Activations with biased distributions (e.g., ReLU outputs, all positive)")
print("    Advantages: More accurate for non-symmetric distributions")
print("    Common use: Activation quantization in QAT")
print()
print("  In practice:")
print("    Weights: Usually symmetric (distributions are roughly symmetric)")
print("    Activations: Often asymmetric (e.g., after ReLU, all values >= 0)")

print("\n[Exercise 3 Answer]")
print("  PTQ (Post-Training Quantization):")
print("    Pros:")
print("      + No retraining needed")
print("      + Fast (minutes to hours)")
print("      + Simple to implement")
print("    Cons:")
print("      - 1-5% accuracy loss")
print("      - May fail at very low precision (INT2-INT3)")
print()
print("  QAT (Quantization-Aware Training):")
print("    Pros:")
print("      + Minimal accuracy loss (< 1%)")
print("      + Works at lower precision")
print("    Cons:")
print("      - Requires full training")
print("      - Slow (hours to days)")
print("      - Complex implementation")
print()
print("  Recommendation:")
print("    INT8: PTQ is sufficient")
print("    INT4: PTQ (GPTQ/AWQ) for most cases, QAT for critical applications")

print("\n[Exercise 4 Answer]")
print("  Per-group quantization is more accurate because:")
print()
print("  Per-tensor:")
print("    One scale for the entire tensor")
print("    -> Outliers (extreme values) determine the scale")
print("    -> Most values get compressed to a narrow range")
print("    -> Example: If one weight is 10.0 and others are ~0.1,")
print("       scale = 10.0/127 ≈ 0.079, most values round to 0 or ±1")
print()
print("  Per-group (e.g., group_size=128):")
print("    Each group of 128 elements gets its own scale")
print("    -> Outliers only affect their own group")
print("    -> Other groups maintain fine-grained quantization")
print("    -> Much better accuracy with only slightly more storage (for scales)")
print()
print("  Storage overhead:")
print("    Per-tensor: 1 scale value")
print("    Per-group: (tensor_size / group_size) scale values")
print("    -> For group_size=128, overhead is < 1% of total storage")

print("\n[Exercise 5 Answer]")
print("  GPTQ's key innovation: Hessian-guided error compensation")
print()
print("  Standard PTQ:")
print("    Quantize each weight independently")
print("    -> Rounding errors accumulate across layers")
print("    -> Output error grows with model depth")
print()
print("  GPTQ:")
print("    1. Compute Hessian H = X^T X (captures input correlations)")
print("    2. Quantize weights column by column")
print("    3. After quantizing column i, compute error e_i = w_i - w_q_i")
print("    4. Adjust remaining columns: w_j -= e_i * H_inv[i,j] / H_inv[i,i]")
print("    5. This compensates for the error in column i by adjusting others")
print()
print("  Result:")
print("    Error doesn't accumulate")
print("    Output is as close as possible to the original")
print("    INT4 perplexity close to FP16")

print("\n[Exercise 6 Answer]")
print("  Deployment tool selection:")
print()
print("  a) Personal laptop:")
print("    -> llama.cpp (GGUF format)")
print("    -> Supports CPU inference, no GPU needed")
print("    -> Various quantization levels (Q4_K_M is a good default)")
print()
print("  b) Serving 100 concurrent users:")
print("    -> vLLM")
print("    -> PagedAttention for efficient memory management")
print("    -> Continuous batching for high throughput")
print("    -> 10-20x throughput vs naive serving")
print()
print("  c) Maximum throughput on NVIDIA GPU:")
print("    -> TensorRT-LLM")
print("    -> NVIDIA's official optimization")
print("    -> FP8/INT4 kernel-level optimization")
print("    -> Best single-GPU throughput")
