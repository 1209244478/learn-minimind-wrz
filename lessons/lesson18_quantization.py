"""
第18课：量化与部署 — 让大模型跑在普通设备上
==============================================

问题: 7B 模型 fp32 权重 28GB, 怎么在 16GB 显存的家用显卡跑？

量化 (Quantization):
  把 fp32 (4字节) 的参数转换为低精度 (2字节/1字节)
  - fp32 → fp16: 2x 压缩
  - fp32 → int8: 4x 压缩
  - fp32 → int4: 8x 压缩
  - 模型体积小, 推理快, 但精度略降

常用方案:
  - PTQ (训练后量化): 直接量化训练好的模型
  - QAT (量化感知训练): 训练时就考虑量化
  - GPTQ, AWQ, GGUF: 不同算法

本课会讲解:
  1. 量化基础 (对称/非对称、per-tensor/per-channel)
  2. 核心算法 (MinMax、GPTQ 简化)
  3. 推理部署 (ONNX、TensorRT)

运行: python lessons/lesson18_quantization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 第1部分：为什么需要量化
# ============================================================

print("=" * 60)
print("第1部分：为什么需要量化")
print("=" * 60)

print("""
模型的"内存墙"问题:
  - 7B 模型 fp32: 28GB 权重
  - 需要至少 2x 显存 (权重 + 激活) ≈ 56GB
  - 消费级显卡 RTX 4090 24GB, 根本放不下

量化的好处:
  ✓ 减少 2-8x 显存占用
  ✓ 推理速度提升 (整数运算比浮点快)
  ✓ 能耗降低 (移动设备友好)
  ✓ 部署门槛低 (CPU 也能跑)

量化的代价:
  ✗ 精度损失 (1-5% perplexity 上升)
  ✗ 训练时无法直接用 (需要 QAT)
  ✗ 不同层可能有不同敏感度
""")

# 演示不同精度的存储
print("\n[演示] 不同精度的存储大小")
print("-" * 60)
print(f"{'精度':<10}{'位数':<8}{'7B 模型权重':<15}{'压缩比':<10}")
print("-" * 60)
for name, bits, factor in [("FP32", 32, 1), ("FP16/BF16", 16, 2), ("INT8", 8, 4), ("INT4", 4, 8)]:
    size_gb = 7 * 4 / factor
    print(f"{name:<10}{bits:<8}{size_gb:<15.1f}{factor}x")


# ============================================================
# 第2部分：量化基础
# ============================================================

print("\n" + "=" * 60)
print("第2部分：量化基础")
print("=" * 60)

print("""
量化的核心思想: 把连续的浮点值映射到离散的整数

两种方案:

1) 对称量化 (Symmetric):
   假设数据分布关于 0 对称
   x_int = round(x / scale)         scale = max(|x|) / (qmax - qmin)
   x_dequant = x_int * scale

   优点: 简单, 硬件友好
   缺点: 数据有偏时浪费量化点

2) 非对称量化 (Asymmetric):
   考虑数据的最小/最大值
   scale = (max - min) / (qmax - qmin)
   zero_point = round(qmin - min / scale)
   x_int = round(x / scale) + zero_point

   优点: 精度更高
   缺点: 复杂一点

粒度:

1) Per-tensor: 整个张量一个 scale/zero_point
2) Per-channel: 每个 channel 一个 scale/zero_point (更准)
3) Per-group: 每 group (32/64) 一个 scale (GPTQ 用)
""")


# ============================================================
# 第3部分：对称量化实现
# ============================================================

print("\n" + "=" * 60)
print("第3部分：对称量化实现")
print("=" * 60)


def quantize_per_tensor_symmetric(tensor, n_bits=8):
    """Per-tensor 对称量化"""
    qmax = 2 ** (n_bits - 1) - 1  # 例如 8-bit: 127
    qmin = -2 ** (n_bits - 1)     # 例如 8-bit: -128

    abs_max = tensor.abs().max()
    scale = abs_max / qmax if abs_max > 0 else 1.0

    tensor_int = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)
    return tensor_int, scale.item()


def dequantize_per_tensor_symmetric(tensor_int, scale, dtype=torch.float32):
    """Per-tensor 对称反量化"""
    return tensor_int.to(dtype) * scale


# 演示对称量化
print("\n[演示] 对称量化示例")
print("-" * 60)
print("原始权重 (FP32):")
W = torch.tensor([[-1.5, 0.2, 0.8], [0.3, -2.1, 1.7], [0.1, 0.5, -0.9]])
print(W)
print(f"  占用字节: {W.numel() * 4}")

W_int8, scale = quantize_per_tensor_symmetric(W, n_bits=8)
print(f"\n量化后 (INT8), scale={scale:.4f}:")
print(W_int8)
print(f"  占用字节: {W_int8.numel() * 1}")

# 反量化
W_dequant = dequantize_per_tensor_symmetric(W_int8, scale)
print(f"\n反量化:")
print(W_dequant)

# 误差
error = (W - W_dequant).abs()
print(f"\n最大绝对误差: {error.max():.4f}")
print(f"平均相对误差: {(error / (W.abs() + 1e-8)).mean():.2%}")


# ============================================================
# 第4部分：GPTQ 简化版 (按组量化)
# ============================================================

print("\n" + "=" * 60)
print("第4部分：GPTQ 简化版 (按组量化)")
print("=" * 60)

print("""
GPTQ 核心思想:
  - 按 group (32/64/128) 量化
  - 用 Hessian 矩阵指导量化, 最小化输出误差
  - 逐列量化, 每列量化后调整剩余列补偿误差

简化版流程:
  1. 计算 Hessian: H = X^T·X
  2. 初始化 errors = 0
  3. for each column i:
     a. 量化第 i 列: w_q = quant(w)
     b. 计算量化误差: e = w - w_q
     c. 更新后续列: w[j] -= e * (H^-1)[i, j] / (H^-1)[i, i]
     d. 累加 error
""")


def quantize_per_group(W, group_size=32, n_bits=4):
    """Per-group 量化 (GPTQ 风格)"""
    out_features, in_features = W.shape
    qmax = 2 ** (n_bits - 1) - 1

    W_int = torch.zeros_like(W, dtype=torch.int8)
    scales = torch.zeros((out_features, in_features // group_size))

    for i in range(0, in_features, group_size):
        w_group = W[:, i:i+group_size]
        # 每组一个 scale (per-row)
        abs_max = w_group.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = abs_max / qmax
        w_int = torch.round(w_group / scale).clamp(-qmax-1, qmax).to(torch.int8)
        W_int[:, i:i+group_size] = w_int
        scales[:, i // group_size] = scale.squeeze()

    return W_int, scales


# 演示 group 量化
print("\n[演示] Per-group 量化")
print("-" * 60)
W = torch.randn(4, 128) * 0.5

W_int, scales = quantize_per_group(W, group_size=32, n_bits=4)
print(f"原始: {W.shape} ({W.numel() * 4} bytes)")
print(f"量化: W_int {W_int.shape} ({W_int.numel() * 1} bytes) + scales {scales.shape} ({scales.numel() * 4} bytes)")
print(f"压缩比: {(W.numel() * 4) / (W_int.numel() * 1 + scales.numel() * 4):.1f}x")


# ============================================================
# 第5部分：Weight-Only 量化 (WnA16)
# ============================================================

print("\n" + "=" * 60)
print("第5部分：Weight-Only 量化")
print("=" * 60)

print("""
Weight-Only 量化 (LLM 部署主流方案):
  - 权重量化为 INT4/INT8
  - 激活保持 FP16
  - 推理时: 动态反量化权重为 FP16, 正常计算

为什么 Weight-Only?
  - LLM 推理是 memory-bound, 计算不是瓶颈
  - 减少权重读取就能加速
  - 激活保持 FP16, 精度损失小

性能对比 (RTX 4090, 7B 模型):
  ┌────────────┬──────────┬──────────┬──────────┐
  │ 精度        │ 显存     │ 吞吐     │ 精度损失  │
  ├────────────┼──────────┼──────────┼──────────┤
  │ FP16       │ 14GB     │ 基准     │ 0%       │
  │ INT8       │ 7.5GB    │ 1.5x     │ <0.5%    │
  │ INT4       │ 4.2GB    │ 2.5x     │ 1-2%     │
  └────────────┴──────────┴──────────┴──────────┘
""")


# ============================================================
# 第6部分：在 MiniMind 中量化
# ============================================================

print("\n" + "=" * 60)
print("第6部分：在 MiniMind 中量化")
print("=" * 60)

print("""
使用 AutoGPTQ 进行 INT4 量化:

```bash
pip install auto-gptq
```

```python
from auto_gptq import BaseQuantizeConfig, AutoGPTQForCausalLM
from transformers import AutoTokenizer

# 加载模型
model_path = "minimind"
quantized_model_path = "minimind-int4"

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,                  # 4-bit 量化
    group_size=128,          # group 大小
    desc_act=True,           # 激活感知
    sym=False,               # 非对称
)

# 准备校准数据
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载并量化
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
calibration_data = [...]  # 校准样本
model.quantize(calibration_data)

# 保存
model.save_quantized(quantized_model_path)
```

使用 bitsandbytes (Transformers 内置):

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # INT4
    bnb_4bit_quant_type="nf4",               # NormalFloat4
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
# 第7部分：部署工具
# ============================================================

print("\n" + "=" * 60)
print("第7部分：部署工具对比")
print("=" * 60)

print("""
主流部署方案:

1) llama.cpp / GGUF
   - 纯 C++ 实现, 性能极高
   - 支持 CPU/GPU
   - 量化格式丰富 (Q2_K, Q4_K_M, Q5_K_M, Q8_0)
   - 适合个人电脑部署

   命令:  ./llama.cpp/main -m model.gguf -p "你好"

2) vLLM
   - PagedAttention 显存管理
   - 吞吐量高 10-20x
   - 适合服务端部署

   ```python
   from vllm import LLM, SamplingParams
   llm = LLM(model="minimind-int4", quantization="gptq")
   ```

3) TensorRT-LLM
   - NVIDIA 官方优化
   - FP8/INT4 极致优化
   - 需要 NVIDIA GPU

4) ONNX Runtime
   - 跨平台
   - 集成到各种应用
   - 性能中等
""")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. 量化基础
   - 浮点 → 整数转换
   - 对称 vs 非对称
   - Per-tensor vs Per-group

2. 主要算法
   - MinMax: 最简单, 性能一般
   - GPTQ: Hessian 引导, SOTA
   - AWQ: 激活感知, 适合 LLM
   - NF4: bitsandbytes 用, 4-bit 友好

3. LLM 部署
   - Weight-Only 量化是主流
   - INT4 + FP16 激活 是最佳平衡
   - 显存节省 3-4x, 精度损失 <2%

4. 部署工具
   - llama.cpp: 个人电脑首选
   - vLLM: 服务端首选
   - TensorRT-LLM: NVIDIA GPU 最优

5. 在 MiniMind 中
   - AutoGPTQ: 最常用
   - bitsandbytes: 最简单
   - 量化后部署到 llama.cpp

下一步: 第19课 - 投机解码加速生成
""")


# ============================================================
# 深入理解：量化的"压缩照片"艺术
# ============================================================
print("\n" + "=" * 60)
print("深入理解：量化的'压缩照片'艺术")
print("=" * 60)

print("""
【类比：量化 = 照片压缩】
────────────────────
  
  原始照片 (FP32):
    10000×10000 像素, 每像素 32 bit
    → 完美, 400 MB
  
  压缩照片 (INT8):
    10000×10000 像素, 每像素 8 bit
    → 几乎一样, 100 MB (4倍压缩)
  
  高度压缩 (INT4):
    每像素 4 bit
    → 略糊, 50 MB (8倍压缩)
  
  极致压缩 (INT2):
    每像素 2 bit
    → 马赛克, 25 MB (16倍压缩)
  
  LLM 量化:
    模型参数当成"像素"
    量化精度当成"压缩比"
    → 用更少空间存模型


【为什么要量化】
────────────

  1. 显存压力:
    LLaMA-70B (FP32): 280 GB → 没法用
    LLaMA-70B (FP16): 140 GB → 需要多卡
    LLaMA-70B (INT4): 35 GB → 单卡可跑!
  
  2. 推理速度:
    内存带宽是瓶颈 (不是计算)
    量化减少数据搬运
    → INT4 比 FP16 快 2-4 倍
  
  3. 部署成本:
    云端 GPU 贵
    量化后用更小的卡
    → 成本降低 50%+
  
  4. 边缘部署:
    手机 / 嵌入式设备
    内存有限, 量化必须
    → 4-bit 模型塞进手机


【量化精度对比】
─────────────

  ┌────────┬────────┬────────┬────────┬──────────┐
  │ 精度    │ 位数   │ 7B 显存 │ 精度损失 │ 速度      │
  ├────────┼────────┼────────┼────────┼──────────┤
  │ FP32   │ 32 bit │ 28 GB  │ 0%     │ 1x        │
  │ FP16   │ 16 bit │ 14 GB  │ ~0%   │ 1.5-2x    │
  │ BF16   │ 16 bit │ 14 GB  │ ~0%   │ 1.5-2x    │
  │ INT8   │ 8 bit  │ 7 GB   │ <1%   │ 1.5-2x    │
  │ INT4   │ 4 bit  │ 3.5 GB │ 2-5%  │ 2-4x      │
  │ INT3   │ 3 bit  │ 2.6 GB │ 5-10% │ 3-5x      │
  │ INT2   │ 2 bit  │ 1.8 GB │ 20%+  │ 5-10x     │
  └────────┴────────┴────────┴────────┴──────────┘
  
  性价比甜蜜点:
    INT4 / INT8 几乎无损失, 性价比最高
    INT2 损失大, 慎用


【量化方法分类】
──────────────

  ┌────────────┬────────────┬────────────┬────────────┐
  │ 类别        │ 精度        │ 速度        │ 难度        │
  ├────────────┼────────────┼────────────┼────────────┤
  │ PTQ        │ 8/4 bit    │ 极快        │ 简单        │
  │ QAT        │ 8/4 bit    │ 慢          │ 复杂        │
  │ GPTQ       │ 4 bit      │ 快          │ 中等        │
  │ AWQ        │ 4 bit      │ 快          │ 中等        │
  │ bitsandbyte│ 4/8 bit    │ 中          │ 简单        │
  └────────────┴────────────┴────────────┴────────────┘

  PTQ (训练后量化):
    直接量化, 不重新训练
    → 几分钟搞定
    → 精度损失 1-5%
  
  QAT (量化感知训练):
    训练时模拟量化
    → 需要完整训练
    → 精度损失 < 1%
  
  GPTQ (最流行):
    按列量化, 补偿误差
    → 几十分钟
    → INT4 几乎无损失
  
  AWQ:
    保护重要权重, 量化次要的
    → 比 GPTQ 略慢
    → 效果略好


【量化的数学原理】
───────────────

  对称量化:
    把 [-max, max] 映射到 [-127, 127] (INT8)
    scale = max / 127
    quantized = round(x / scale)
    
    反量化:
    dequantized = quantized * scale
  
  非对称量化:
    把 [min, max] 映射到 [0, 255] (UINT8)
    scale = (max - min) / 255
    zero_point = round(-min / scale)
    quantized = round(x / scale) + zero_point
    
    反量化:
    dequantized = (quantized - zero_point) * scale
  
  例子 (INT8 对称):
    权重: [0.1, 0.5, -0.3, 0.8]
    max = 0.8
    scale = 0.8 / 127 ≈ 0.0063
    量化: [16, 79, -48, 127]
    反量化: [0.101, 0.498, -0.302, 0.799]
    误差: < 0.01 (0.8%)


【Per-Channel vs Per-Tensor】
──────────────────────────

  Per-Tensor (粗粒度):
    整个 tensor 共用一个 scale
    → 简单, 但精度差
    → 极端值会"压扁"其他值
  
  Per-Channel (细粒度):
    每个 channel 独立 scale
    → 复杂, 但精度好
    → 工业界默认
  
  Per-Group (更细):
    每 32/64/128 个元素一组
    → GPTQ / AWQ 用
    → 精度最好


【量化误差来源】
─────────────

  1. 舍入误差:
    真实值 0.123 → INT4 没有 0.123 这个级别
    → 量化到 0.125, 误差 0.002
    → 大部分参数, 误差累积
  
  2. 异常值:
    大部分权重在 [-0.1, 0.1]
    少数权重达到 ±5
    → max=5 决定 scale
    → 大部分被压到 0 附近, 精度丢失
  
  解决:
    - 裁剪异常值 (clamp)
    - 单独处理异常值通道 (AWQ)
    - 分组量化 (GPTQ)


【GPTQ 的核心思想】
─────────────────

  问题: 逐层量化误差累积
  
  解决: 用二阶信息 (Hessian 矩阵) 补偿
  
  步骤:
    1. 计算 H = X^T X (输入的协方差)
    2. 逐列量化权重
    3. 每量化一列, 调整未量化的列来补偿误差
    4. → 误差不累积, 精度高
  
  复杂度:
    一次校准数据 (几百样本)
    → 几十分钟完成
  
  效果:
    INT4 LLaMA-70B:
      普通 PTQ:  perplexity 8+
      GPTQ:      perplexity 5.5
      (接近 FP16 的 5.4)


【AWQ 的核心思想】
────────────────

  观察:
    不是所有权重都重要
    0.1-1% 的"显著权重"决定输出
    
  策略:
    1. 识别显著权重 (激活值大)
    2. 不量化这些, 保持 FP16
    3. 量化其他权重到 INT4
    
  优势:
    显著权重决定质量
    量化其他权重省空间
    → 精度损失极小
    
  对比:
    GPTQ: 改权重值
    AWQ:  改"哪些权重量化"


【混合精度量化】
──────────────

  思想: 不同层用不同精度
  
  策略:
    - Embedding 层: FP16 (重要, 占用大)
    - Attention 层: INT4 (重复结构)
    - FFN 层: INT4 (参数量大)
    - LM Head: FP16 (输出关键)
  
  效果:
    总精度 INT4.5
    关键层保留精度
    → 性能接近 INT8, 显存接近 INT4


【量化 vs 蒸馏 vs 剪枝】
─────────────────────

  ┌──────────┬────────────┬────────────┬────────────┐
  │ 技术      │ 原理        │ 速度        │ 精度保持    │
  ├──────────┼────────────┼────────────┼────────────┤
  │ 量化     │ 降精度      │ 极快        │ 95-99%     │
  │ 蒸馏     │ 小模型学大   │ 慢 (训练)   │ 90-95%     │
  │ 剪枝     │ 去冗余      │ 快          │ 90-95%     │
  └──────────┴────────────┴────────────┴────────────┘

  量化优势:
    - 不需要重新训练
    - 几分钟搞定
    - 精度保持好
    → 工业界首选


【部署框架】
──────────

  ┌──────────────┬────────────┬──────────────────┐
  │ 框架          │ 优势        │ 适用              │
  ├──────────────┼────────────┼──────────────────┤
  │ llama.cpp    │ 跨平台 CPU  │ 边缘/本地        │
  │ vLLM         │ 服务端      │ 高吞吐服务        │
  │ TensorRT-LLM │ NVIDIA 最优 │ 生产 GPU 服务     │
  │ TGI          │ HuggingFace │ 云端服务          │
  │ MLC-LLM      │ 移动/Web   │ 手机/浏览器      │
  │ ONNX Runtime │ 跨平台      │ 多平台部署        │
  └──────────────┴────────────┴──────────────────┘
  
  推荐:
    本地体验: llama.cpp (CPU/GPU 都行)
    服务部署: vLLM (吞吐量最高)
    极致性能: TensorRT-LLM
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】量化收益
  LLaMA-7B FP16 需要多少显存? INT4 呢?

【练习2】对称 vs 非对称
  对称量化和非对称量化各适合什么场景?

【练习3】PTQ vs QAT
  训练后量化 (PTQ) 和量化感知训练 (QAT) 各自优缺点?

【练习4】GPTQ 核心
  GPTQ 为什么用 Hessian 矩阵? 它做了什么?

【练习5】异常值问题
  权重有 99% 在 [-0.1, 0.1], 1% 在 [-5, 5]
  INT8 量化会有什么问题? 怎么解决?

【练习6】量化选择
  你的 7B 模型要部署到手机, 应该选什么量化方案?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  LLaMA-7B 显存计算:")
print()
print("  模型参数: 7B")
print("  FP32:  7B × 4 字节 = 28 GB")
print("  FP16:  7B × 2 字节 = 14 GB")
print("  INT8:  7B × 1 字节 = 7 GB")
print("  INT4:  7B × 0.5 字节 = 3.5 GB")
print()
print("  实际部署时还要考虑:")
print("    KV Cache: 1-2 GB (上下文相关)")
print("    激活值: 1-5 GB (batch 相关)")
print("    框架开销: 0.5-1 GB")
print()
print("  INT4 部署:")
print("    总显存: 3.5 + 2 + 1 = 6.5 GB")
print("    → 单卡 RTX 3060 (12GB) 即可")
print("    → 甚至可以上 8GB 显卡")
print()
print("  对比:")
print("    FP16: 14 GB → 需要 16GB+ 显卡")
print("    INT4: 3.5 GB → 8GB 显卡即可")

# 练习2
print("\n【练习2 答案】")
print("  对称量化 (Symmetric):")
print("    范围: [-max, max]")
print("    优点: 计算简单, 硬件友好")
print("    缺点: 范围不均衡时浪费精度")
print("    适用: 权重 (分布大致对称)")
print()
print("  非对称量化 (Asymmetric):")
print("    范围: [min, max]")
print("    优点: 充分利用 INT8 范围")
print("    缺点: 多了一个 zero_point")
print("    适用: 激活值 (分布可能偏斜)")
print()
print("  实际选择:")
print("    权重量化: 多用对称 (速度快)")
print("    激活量化: 多用非对称 (精度好)")
print("    工业框架: 二者都支持, 自动选")
print()
print("  INT4 常用:")
print("    对称 (因为 INT4 只有 16 个值, 不对称优势小)")

# 练习3
print("\n【练习3 答案】")
print("  PTQ (Post-Training Quantization):")
print()
print("  优点:")
print("    - 速度快 (几十分钟)")
print("    - 不需要训练数据")
print("    - 实现简单")
print("    - 不需要修改训练流程")
print()
print("  缺点:")
print("    - 精度损失较大 (1-5%)")
print("    - 对异常值敏感")
print("    - 极端量化 (INT2) 难以用")
print()
print("  适用:")
print("    - 快速部署")
print("    - 资源有限")
print("    - 模型已收敛")
print()
print("  ===")
print()
print("  QAT (Quantization-Aware Training):")
print()
print("  优点:")
print("    - 精度损失小 (< 1%)")
print("    - 极端量化也行")
print("    - 模型适应量化")
print()
print("  缺点:")
print("    - 需要完整训练流程")
print("    - 训练时间长 (数天)")
print("    - 需要训练数据")
print("    - 调参复杂")
print()
print("  适用:")
print("    - 极致精度")
print("    - 极端量化")
print("    - 长期项目")
print()
print("  实际选择:")
print("    一般: PTQ (GPTQ, AWQ)")
print("    极致: QAT")

# 练习4
print("\n【练习4 答案】")
print("  GPTQ 用 Hessian 矩阵的原因:")
print()
print("  朴素量化的问题:")
print("    逐列量化, 误差累积")
print("    → 后面的列补偿前面的误差")
print("    → 误差无法控制")
print()
print("  GPTQ 的洞察:")
print("    误差的影响不是独立的")
print("    → 与输入分布有关")
print("    → 用 Hessian H = X^T X 描述这种关系")
print()
print("  GPTQ 步骤:")
print("    1. 用校准数据计算 H")
print("    2. 对 Hessian 做 Cholesky 分解")
print("    3. 逐列量化权重")
print("    4. 量化 w_j 后, 用 H 的列调整未量化列")
print("       w_i = w_i - (H_ij / H_jj) * (quant(w_j) - w_j)")
print("    5. → 后续列补偿前面误差")
print()
print("  数学基础:")
print("    最小化 ||Wx - W_q x||² 在 X 上的期望")
print("    → 最优解需要 H 的信息")
print()
print("  效果:")
print("    误差不再累积")
print("    → INT4 perplexity 接近 FP16")

# 练习5
print("\n【练习5 答案】")
print("  问题分析:")
print()
print("  权重分布:")
print("    99% 在 [-0.1, 0.1]   → 主体")
print("    1% 在 [-5, 5]        → 异常值")
print()
print("  INT8 量化 (Per-Tensor):")
print("    max = 5 (被异常值决定)")
print("    scale = 5 / 127 ≈ 0.039")
print("    99% 权重 / 0.039 = [-2.5, 2.5]")
print("    → 量化到 [-2, 2] 范围")
print("    → 大量相邻值被压到同一个 INT8 值")
print("    → 精度严重丢失")
print()
print("  后果:")
print("    主体权重 (大多数参数) 几乎无法区分")
print("    → 模型质量大幅下降")
print()
print("  解决方案:")
print("  1. Per-Channel 量化:")
print("    每通道独立 scale")
print("    → 异常值通道单独用大 scale")
print("    → 主体通道用小 scale, 精度好")
print()
print("  2. 裁剪 (Clipping):")
print("    把异常值裁到 [-1, 1]")
print("    → max = 1, scale = 1/127")
print("    → 主体权重精度好")
print("    → 异常值信息有损")
print()
print("  3. AWQ:")
print("    不量化异常值, 保留 FP16")
print("    → 异常值是'显著权重', 重要")
print("    → 其他用 INT4 量化")
print()
print("  4. SmoothQuant:")
print("    把量化难度从激活转移到权重")
print("    → 激活值异常值用 scale 平滑")
print("    → 权重相应缩放")
print("    → 双向都好")
print()
print("  5. 分组量化 (GPTQ):")
print("    每 32/64/128 元素一个 scale")
print("    → 异常值只影响所在组")
print("    → 工业标准做法")

# 练习6
print("\n【练习6 答案】")
print("  7B 模型部署到手机的方案:")
print()
print("  手机硬件:")
print("    RAM: 4-8 GB")
print("    存储: 可用 2-4 GB 给模型")
print("    芯片: ARM CPU / 部分有 NPU")
print()
print("  量化方案选择:")
print()
print("  推荐: INT4 量化")
print("    模型大小: 7B × 0.5 字节 ≈ 3.5 GB")
print("    加上框架开销: ~4 GB")
print("    → 大部分手机能跑")
print()
print("  具体工具:")
print("    1. llama.cpp + 4-bit 量化")
print("      → 最成熟, 跨平台")
print("      → 手机 CPU 上能跑")
print("      → 速度: 几 token/秒")
print()
print("    2. MLC-LLM")
print("      → 专为移动端优化")
print("      → 支持 GPU/NPU 加速")
print("      → 速度: 10+ token/秒")
print()
print("    3. llama.cpp + Q4_K_M")
print("      → 4-bit, 中等质量")
print("      → 平衡速度和精度")
print()
print("  量化方法选择:")
print("    GPTQ: 容易集成")
print("    AWQ:  速度快")
print("    bitsandbytes: 最简单")
print()
print("  性能预期 (iPhone 14):")
print("    7B INT4: 5-15 token/秒")
print("    3B INT4: 15-30 token/秒")
print("    1.5B INT4: 30+ token/秒")
print()
print("  优化建议:")
print("    - 优先用 NPU 加速")
print("    - 选 ARM NEON 优化版本")
print("    - 用 KV Cache 节省内存")
print("    - 量化激活到 INT8 进一步省")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
