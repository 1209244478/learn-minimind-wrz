"""
第27课：LLM 面试进阶 — 高频面试题全解析
=============================================

本课汇总了大模型领域的高频面试题，覆盖：
  1. 基础架构篇 — Transformer 核心机制
  2. 训练对齐篇 — 预训练/SFT/RLHF/DPO
  3. 推理优化篇 — KV Cache / 量化 / 投机解码
  4. 前沿技术篇 — MoE / Mamba / GRPO / 多模态
  5. 工程实战篇 — 分布式训练 / 部署 / 长上下文
  6. 数学推导篇 — 手推核心公式

每道题都包含：问题 -> 核心答案 -> 深度追问 -> 代码验证（如有）

运行: python lessons/lesson27_interview.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 第1部分：基础架构篇
# ============================================================

print("=" * 70)
print("第1部分：基础架构篇 — Transformer 核心机制")
print("=" * 70)

print("""
============================================================
Q1: Transformer 为什么用 Self-Attention 而不用 RNN？
============================================================

核心答案：
  1. 并行计算：RNN 必须顺序处理（t 时刻依赖 t-1），Attention 可以同时计算所有位置
  2. 长距离依赖：RNN 经过多次传播梯度消失，Attention 任意两位置距离为 O(1)
  3. 计算复杂度：
     - RNN: O(n*d^2)  — n 是序列长度，d 是维度
     - Attention: O(n^2*d) — 序列平方，但可并行
     - 当 n < d^2/n 时（短序列），Attention 更快

深度追问：
  Q: 那 Attention 的 O(n^2)) 复杂度怎么解决？
  A: 多种方案：
     - Flash Attention: 不降低复杂度，但减少 HBM 访问（IO-aware），实际快2-4x
     - 稀疏 Attention: 只看部分位置（Longformer, BigBird）
     - 线性 Attention: 用核函数近似 softmax（Performers）
     - 状态空间模型: O(n) 复杂度（Mamba, S4）

============================================================
Q2: 为什么 Transformer 要用 Multi-Head Attention？
============================================================

核心答案：
  单头 Attention 只能学一种"关注模式"（比如只看语法关系），
  多头让模型同时关注不同类型的关系：
    - 头1：关注语法关系（主语->谓语）
    - 头2：关注指代关系（代词->名词）
    - 头3：关注位置关系（相邻词）
    - 头4：关注语义关系（同义词）

  数学上：MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_o
  每个头的维度 = d_model / n_heads，总计算量不变

深度追问：
  Q: 为什么不直接用一个大头（比如 d=512 的单头）？
  A: 实验表明多头效果更好。类比：8个专家各看一个角度 > 1个通才看所有角度。
     大头容易学出"平均"模式，多头能学出"多样化"模式。

============================================================
Q3: LayerNorm vs BatchNorm？为什么 Transformer 用 LayerNorm？
============================================================

核心答案：
  BatchNorm: 沿 batch 维度归一化 -> 每个 feature 跨样本归一化
  LayerNorm: 沿 feature 维度归一化 -> 每个样本内部归一化

  Transformer 用 LayerNorm 的原因：
  1. 序列长度可变 -> BatchNorm 统计量不稳定
  2. Batch size 小时（如推理时 bs=1）-> BatchNorm 退化
  3. 自回归生成时，序列逐步增长 -> BatchNorm 无法处理
  4. LayerNorm 对每个 token 独立归一化，不受 batch 影响

  更进一步：RMSNorm vs LayerNorm
    LayerNorm: y = (x - mu) / sigma * gamma + beta    （减均值 + 除标准差 + 缩放 + 偏移）
    RMSNorm:   y = x / RMS(x) * gamma          （只除RMS + 缩放，不减均值不加偏移）
    RMSNorm 省掉了均值和偏移计算，速度快 ~10%，效果基本相同

============================================================
Q4: 残差连接（Residual Connection）为什么有效？
============================================================

核心答案：
  1. 梯度直通：反向传播时，梯度可以通过残差路径直接传到前面
     y = x + F(x) -> dL/dx = dL/dy * (1 + dF/dx)
     即使 dF/dx 很小，1 保证了梯度不会消失

  2. 信息保留：每层只需要学"增量"（残差），不用重新学全部信息
     类比：写论文时在初稿上修改（残差）vs 从头重写（无残差）

  3. 恒等映射初始化：如果 F(x)=0，y=x，至少不会比不加深网络差

深度追问：
  Q: Pre-Norm vs Post-Norm 哪个好？
  A: Pre-Norm（先 LayerNorm 再 Attention）更稳定，训练不需要 warmup
     Post-Norm（先 Attention 再 LayerNorm）效果可能更好但训练不稳定
     目前主流用 Pre-Norm（GPT、LLaMA 都用 Pre-Norm）

============================================================
Q5: 位置编码为什么重要？RoPE 的核心思想是什么？
============================================================

核心答案：
  Self-Attention 是位置无关的（置换不变性）：
    Attention(["我", "爱", "你"]) = Attention(["你", "爱", "我"])
  所以必须注入位置信息。

  RoPE（旋转位置编码）的核心思想：
    用旋转矩阵编码相对位置，使得 q*k 的点积只依赖相对距离

  具体做法：将 q 和 k 的相邻维度配对，旋转角度 = 位置 x 频率
    q_m = [q_0 cos(mtheta_0) - q_1 sin(mtheta_0), q_0 sin(mtheta_0) + q_1 cos(mtheta_0), ...]
    k_n = [k_0 cos(ntheta_0) - k_1 sin(ntheta_0), k_0 sin(ntheta_0) + k_1 cos(ntheta_0), ...]
    q_m * k_n = f(m-n)  <- 只依赖相对位置 m-n！

  为什么用不同频率？
    低频（theta小）：旋转慢，适合捕捉长距离位置关系
    高频（theta大）：旋转快，适合捕捉短距离位置关系
    类比时钟：秒针（高频）捕捉秒级变化，时针（低频）捕捉小时级变化
""")

# --- 代码验证：RoPE 相对位置 ---
print("\n[代码验证] RoPE 使 Attention 只依赖相对位置")
print("-" * 50)

d = 4  # 简化：4维
theta = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))

def apply_rope(x, pos):
    """对向量 x 在位置 pos 应用 RoPE"""
    x_pairs = x.float().view(-1, 2)
    cos_val = torch.cos(pos * theta)
    sin_val = torch.sin(pos * theta)
    rotated = torch.stack([
        x_pairs[:, 0] * cos_val - x_pairs[:, 1] * sin_val,
        x_pairs[:, 0] * sin_val + x_pairs[:, 1] * cos_val
    ], dim=-1)
    return rotated.view(-1)

q = torch.randn(d)
k = torch.randn(d)

# 计算不同位置组合的点积
for m in range(4):
    for n in range(4):
        dot = torch.dot(apply_rope(q, m), apply_rope(k, n))
        if m - n == 1:  # 相对距离为1
            print(f"  位置({m},{n}) 相对距离={m-n}: 点积={dot.item():.4f}")

print("  -> 相对距离相同的组合，点积值相同（RoPE 的核心性质）")


# ============================================================
# 第2部分：训练对齐篇
# ============================================================

print("\n" + "=" * 70)
print("第2部分：训练对齐篇 — 预训练/SFT/RLHF/DPO")
print("=" * 70)

print("""
============================================================
Q6: 大模型训练的三个阶段分别做什么？
============================================================

核心答案：
  阶段1 — 预训练（Pre-training）：
    目标：学会语言的统计规律
    数据：海量无标注文本（网页、书籍、代码...）
    任务：下一个 token 预测（Next Token Prediction）
    损失：CrossEntropyLoss
    类比：读完全部维基百科，学会"说话"

  阶段2 — 监督微调（SFT, Supervised Fine-Tuning）：
    目标：学会按指令回答问题
    数据：指令-回答对（"解释量子力学" -> "量子力学是..."）
    任务：条件文本生成
    类比：读完百科后，上"问答课"学会回答问题

  阶段3 — 对齐（Alignment: RLHF / DPO）：
    目标：让回答符合人类偏好（有用、安全、诚实）
    数据：人类偏好标注（回答A比回答B好）
    方法：RLHF（训练奖励模型 + PPO）或 DPO（直接优化偏好）
    类比：学会"说好话"而不是"乱说话"

深度追问：
  Q: 为什么不能跳过预训练直接 SFT？
  A: SFT 数据量太小（几万条 vs 几千亿 token），模型学不到语言基础。
     类比：没上过小学直接上高中，听不懂。

  Q: 为什么需要对齐阶段？
  A: SFT 只教会"怎么回答"，没教会"什么回答是好的"。
     模型可能学会模仿格式但内容有害/无用。
     对齐阶段让模型区分"好回答"和"坏回答"。

============================================================
Q7: DPO 和 RLHF 有什么区别？各有什么优劣？
============================================================

核心答案：
  RLHF 流程：
    1. 训练奖励模型 RM（学习人类偏好）
    2. 用 PPO 优化策略模型（最大化 RM 给的奖励）
    需要：4个模型（策略模型、参考模型、奖励模型、价值模型）
    问题：训练复杂、不稳定、超参多

  DPO 流程：
    直接用偏好数据优化策略模型，跳过奖励模型
    数学推导：
      RLHF 最优解：pi*(y|x) propto pi_ref(y|x) * exp(r(x,y)/beta)
      反解奖励：r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))
      代入 Bradley-Terry 模型，得到 DPO 损失：
      L_DPO = -log sigma(beta * [log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))])

  对比：
    +----------┬--------------┬--------------+
    |          |    RLHF      |     DPO      |
    +----------+--------------+--------------+
    | 模型数量 | 4个          | 2个          |
    | 训练复杂 | 高           | 低           |
    | 稳定性   | 不稳定       | 稳定         |
    | 奖励模型 | 需要         | 不需要       |
    | 在线学习 | 支持         | 不支持       |
    | 效果上限 | 更高(理论上) | 略低         |
    +----------+--------------+--------------┘

深度追问：
  Q: 什么时候用 DPO，什么时候用 RLHF？
  A: 资源有限/快速迭代 -> DPO；追求极致效果/有在线数据 -> RLHF
     实际上很多团队先用 DPO 快速验证，再用 RLHF 精调

============================================================
Q8: GRPO 是什么？和 PPO 有什么区别？
============================================================

核心答案：
  GRPO = Group Relative Policy Optimization（组相对策略优化）
  DeepSeek 提出的简化版 RLHF 方法。

  核心思想：不用价值模型，用"组内相对排名"代替"绝对奖励"

  PPO 的优势函数：A(s,a) = Q(s,a) - V(s)  <- 需要训练价值模型 V
  GRPO 的优势函数：A_i = (r_i - mean(r)) / std(r)  <- 只需要组内统计

  具体做法：
    1. 对同一个问题，采样 G 个回答
    2. 用奖励模型给每个回答打分 r_1, ..., r_G
    3. 计算组内标准化优势：A_i = (r_i - mean) / std
    4. 用优势加权更新策略

  优势：省掉价值模型，训练更简单，DeepSeek-V2/V3 都用 GRPO
""")

# --- 代码验证：GRPO 优势计算 ---
print("\n[代码验证] GRPO 优势计算")
print("-" * 50)

rewards = torch.tensor([[0.8, 0.3, 0.5, 0.1],   # 问题1的4个回答
                         [0.9, 0.7, 0.2, 0.4]])   # 问题2的4个回答
mean = rewards.mean(dim=1, keepdim=True)
std = rewards.std(dim=1, keepdim=True) + 1e-8
advantages = (rewards - mean) / std

print(f"奖励: {rewards.tolist()}")
print(f"组均值: {mean.squeeze().tolist()}")
print(f"组标准差: {std.squeeze().tolist()}")
print(f"GRPO优势: {[[f'{a:.2f}' for a in row] for row in advantages.tolist()]}")
print("-> 正值=比组内平均好，负值=比组内平均差")


# ============================================================
# 第3部分：推理优化篇
# ============================================================

print("\n" + "=" * 70)
print("第3部分：推理优化篇 — KV Cache / 量化 / 投机解码")
print("=" * 70)

print("""
============================================================
Q9: KV Cache 是什么？为什么能加速推理？
============================================================

核心答案：
  自回归生成时，每一步都要算 Attention(q_t, K, V)
  其中 K=[k_1,...,k_t], V=[v_1,...,v_t] 包含了之前所有 token 的信息

  没有 KV Cache：每步重新计算所有 k_1,...,k_t（重复计算！）
  有 KV Cache：只算新的 k_t, v_t，拼到缓存里

  复杂度对比（生成 T 个 token）：
    无 Cache: O(T^2 * d)  — 每步重算所有
    有 Cache: O(T * d)   — 每步只算新的

  内存占用（LLaMA-2-70B 为例）：
    每个token的KV Cache = 2 x n_kv_heads x d_head x seq_len x 2 bytes
    MHA (64 KV heads): ~32 KB/token -> 128MB for 4096 tokens
    GQA (8 KV heads):  ~4 KB/token  -> 16MB for 4096 tokens

深度追问：
  Q: KV Cache 的内存瓶颈怎么解决？
  A: 多种方案：
     - GQA: 减少 KV heads（LLaMA-2 用 8 个代替 64 个）
     - MQA: 所有 head 共享 1 组 KV（极端版 GQA）
     - PagedAttention: 虚拟内存管理，避免碎片（vLLM）
     - KV Cache 量化: FP16 -> INT8/INT4，省一半/四分之一内存
     - 滑动窗口: 只保留最近 W 个 token 的 KV（Mistral）

============================================================
Q10: 模型量化是什么？INT8/INT4 量化怎么做？
============================================================

核心答案：
  量化 = 用更少的比特表示模型参数
  FP32 -> FP16 -> INT8 -> INT4

  两种量化方式：
  1. 训练后量化（PTQ, Post-Training Quantization）：
     直接把训练好的模型参数从 FP16 转成 INT8/INT4
     方法：AbsMax（除以最大绝对值）、MinMax（映射到[-128,127]）

  2. 量化感知训练（QAT, Quantization-Aware Training）：
     训练时模拟量化误差，让模型适应低精度
     效果更好但训练成本高

  精度损失：
    FP16 -> INT8: 几乎无损（<1% 准确率下降）
    FP16 -> INT4: 轻微损失（1-3%），需要 GPTQ/AWQ 等方法补偿

  内存和速度收益：
    7B 模型 FP16: 14GB 显存
    7B 模型 INT8: 7GB 显存
    7B 模型 INT4: 3.5GB 显存（可以在 6GB 显卡上跑！）

============================================================
Q11: 投机解码（Speculative Decoding）是什么？
============================================================

核心答案：
  核心矛盾：大模型生成质量高但慢，小模型快但质量差
  投机解码：用小模型"猜"，大模型"验"

  流程：
    1. 小模型（draft model）快速生成 K 个 token
    2. 大模型（target model）一次前向传播验证这 K 个 token
    3. 接受正确的 token，拒绝错误的，从拒绝点重新生成

  为什么能加速？
    - 大模型一次前向传播可以同时验证 K 个 token
    - 如果小模型猜对率 p，期望接受长度 = 1/(1-p)
    - 小模型猜对率通常 70-90%，所以期望接受 3-10 个 token
    - 加速比：2-3x（无损，输出和纯大模型完全一致）

深度追问：
  Q: 为什么输出和纯大模型完全一致？
  A: 拒绝时，按大模型的概率分布重新采样，数学上等价于直接从大模型采样。
     这是投机解码的精妙之处：加速但不牺牲质量。
""")

# --- 代码验证：量化效果 ---
print("\n[代码验证] 量化对精度的影响")
print("-" * 50)

def quantize_int8(tensor):
    """简单的 AbsMax INT8 量化"""
    scale = tensor.abs().max() / 127
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale

def dequantize_int8(quantized, scale):
    return quantized.float() * scale

original = torch.randn(100)
q8, scale = quantize_int8(original)
recovered = dequantize_int8(q8, scale)
error = (original - recovered).abs().mean()

print(f"原始值范围: [{original.min():.4f}, {original.max():.4f}]")
print(f"INT8 量化后范围: [{q8.min()}, {q8.max()}]")
print(f"反量化后范围: [{recovered.min():.4f}, {recovered.max():.4f}]")
print(f"平均误差: {error:.6f} (相对误差: {error/original.abs().mean()*100:.2f}%)")


# ============================================================
# 第4部分：前沿技术篇
# ============================================================

print("\n" + "=" * 70)
print("第4部分：前沿技术篇 — MoE / Mamba / GRPO / 多模态")
print("=" * 70)

print("""
============================================================
Q12: MoE（混合专家）是什么？有什么优缺点？
============================================================

核心答案：
  MoE = Mixture of Experts，用多个"专家"网络替代单个 FFN

  结构：
    输入 x -> Router（门控网络）-> 选择 Top-K 个专家 -> 加权求和
    Router(x) = softmax(W_r * x)  -> 每个专家的权重
    output = sum gate_i * Expert_i(x)  （只选 Top-K 个）

  优势：
    1. 参数量大但计算量小：8个专家只激活2个，参数量8x但计算量不变
    2. 模型容量大：更多参数 = 更多知识存储
    3. 专业化：不同专家学不同领域的知识

  问题：
    1. 路由崩塌（Router Collapse）：所有 token 都被分给少数专家
       解决：加辅助损失 aux_loss = alpha * n * sum p_i^2  （鼓励均匀分配）
    2. 负载不均：GPU 间专家分配不均，有的 GPU 空闲
       解决：Expert Parallelism + 容量因子（Capacity Factor）
    3. 通信开销：跨 GPU 的 All-to-All 通信
       解决：减少通信频率、用更快的网络

  代表模型：Mixtral 8x7B、DeepSeek-V2/V3、Switch Transformer

深度追问：
  Q: MoE 的"专家"真的学到了不同领域的知识吗？
  A: 研究发现，专家确实有一定专业化，但不像人想象的那样"数学专家""代码专家"。
     更多是"浅层模式"的分化：某些专家处理标点，某些处理特定语法结构。
     完全的领域专业化需要更强的约束（如专家选择损失）。

============================================================
Q13: Mamba/SSM 和 Transformer 有什么区别？
============================================================

核心答案：
  SSM（State Space Model）的核心方程：
    h'(t) = A*h(t) + B*x(t)   （状态更新）
    y(t)  = C*h(t) + D*x(t)   （输出）

  类比：
    Attention = 开会时每个人都能直接和所有人对话（信息丰富但 O(n^2))）
    SSM = 开会时每个人只听前一个人的总结（快速 O(n) 但信息压缩）
    Mamba = SSM + 选择性机制：重要信息多记住，不重要信息少记住

  Mamba 的关键创新 — 选择性机制：
    传统 SSM：A, B, C 是固定的（不管输入是什么，同样的压缩方式）
    Mamba：A, B, C 依赖输入 x（重要信息保留更多，不重要信息压缩更多）
    类比：读书时重要段落做详细笔记，不重要段落只记关键词

  对比：
    +--------------┬--------------┬--------------+
    |              |  Transformer |    Mamba     |
    +--------------+--------------+--------------+
    | 训练复杂度   | O(n^2))        | O(n)         |
    | 推理复杂度   | O(n) per step| O(1) per step|
    | 长序列       | 受限(KV Cache)| 天然支持     |
    | 并行训练     | 好           | 好(并行扫描) |
    | 效果         | 成熟/验证多  | 快速追赶中   |
    +--------------+--------------+--------------┘

深度追问：
  Q: Mamba 会取代 Transformer 吗？
  A: 短期不会。Transformer 生态太成熟，Mamba 还在验证阶段。
     但混合架构（Jamba = Mamba + Attention）可能是未来方向。
     长序列场景（DNA、音频）Mamba 有天然优势。

============================================================
Q14: 多模态大模型（VLM）是怎么把图像和文本对齐的？
============================================================

核心答案：
  核心挑战：图像是2D像素，文本是1D token，怎么让它们"说同一种语言"？

  主流方案：视觉编码器 + 投影层 + 语言模型
    1. 视觉编码器（如 ViT/CLIP）：把图像切成 patch，提取视觉特征
       图像 -> [patch_1, patch_2, ..., patch_N] -> 视觉特征 [v_1, ..., v_N]
    2. 投影层（Projector）：把视觉特征映射到语言模型的嵌入空间
       v_i -> p_i = W_project * v_i  （维度对齐）
    3. 语言模型：把投影后的视觉 token 和文本 token 拼在一起处理
       输入 = [视觉token_1, ..., 视觉token_N, 文本token_1, ...]

  代表模型：
    - LLaVA: CLIP ViT + MLP 投影 + LLaMA
    - Qwen-VL: ViT + Cross-Attention + Qwen
    - GPT-4V: 架构未公开，推测类似方案

  PatchEmbedding 怎么把图像切成 token？
    1. 图像 224x224x3 -> 切成 16x16 的 patch -> 14x14=196 个 patch
    2. 每个 patch 16x16x3=768 维 -> 用线性层映射到 d_model 维
    3. 加上位置编码 -> 得到 196 个视觉 token

深度追问：
  Q: 为什么不直接用像素作为 token？
  A: 224x224=50176 个像素，序列太长（Attention O(n^2)) 扛不住）。
     切成 patch 后只有 196 个 token，可控。
     而且 patch 级别的特征比像素级别更有语义信息。
""")


# ============================================================
# 第5部分：工程实战篇
# ============================================================

print("\n" + "=" * 70)
print("第5部分：工程实战篇 — 分布式训练 / 部署 / 长上下文")
print("=" * 70)

print("""
============================================================
Q15: 分布式训练有哪些并行策略？
============================================================

核心答案：
  1. 数据并行（Data Parallelism, DP）：
     每张卡有完整模型副本，数据分片
     前向->收集梯度->平均->更新
     瓶颈：梯度通信；模型太大单卡放不下

  2. ZeRO（Zero Redundancy Optimizer）：
     DP 的优化版，分片存储优化器状态/梯度/参数
     ZeRO-1: 分片优化器状态 -> 省4x内存
     ZeRO-2: 分片优化器+梯度 -> 省8x内存
     ZeRO-3: 分片优化器+梯度+参数 -> 省N倍（N=GPU数）

  3. 张量并行（Tensor Parallelism, TP）：
     把单个矩阵乘法切分到多卡
     Y = X * W -> 列切分 W=[W1, W2]，Y = [X*W1, X*W2]
     通信量大，适合节点内（NVLink）

  4. 流水线并行（Pipeline Parallelism, PP）：
     模型按层切分到不同卡
     GPU0: Layer 0-7, GPU1: Layer 8-15, ...
     问题：气泡（bubble），GPU 空闲等待
     解决：微批次（micro-batch）填充流水线

  5. 序列并行（Sequence Parallelism, SP）：
     把长序列切分到多卡
     适合超长上下文（128K+ tokens）

  实际组合（以 LLaMA-2-70B 训练为例）：
    TP=4（节点内4卡张量并行）
    x PP=2（2个流水线阶段）
    x DP=64（64路数据并行）
    = 512 张 A100

============================================================
Q16: 长上下文（Long Context）有哪些技术方案？
============================================================

核心答案：
  1. 训练时扩展上下文：
     - 位置插值（Position Interpolation）：压缩位置编码
     - YaRN：按频率分组处理（高频外推、低频插值）
     - NTK-aware 缩放：调整 RoPE 的 base 频率
     - 逐步扩展：先训练 4K，再扩展到 32K，再扩展到 128K

  2. 推理时优化：
     - 滑动窗口 Attention：只看最近 W 个 token
     - 稀疏 Attention：关键 token 全连接，其他局部
     - KV Cache 压缩：丢弃不重要的 KV（如 H2O）
     - 分块预填充（Chunked Prefill）：分批处理长输入

  3. RAG（检索增强生成）：
     不直接把长文档喂给模型，而是先检索相关段落
     适合：知识密集型任务（问答、摘要）
     不适合：需要全文理解的任务（长文档推理）

  代表模型上下文长度：
    GPT-4: 128K tokens
    Claude-3: 200K tokens
    Gemini-1.5: 1M-2M tokens
    LLaMA-3: 8K -> 128K（扩展训练）

============================================================
Q17: 模型部署有哪些方案？怎么选择？
============================================================

核心答案：
  推理框架对比：
    +-----------┬--------------┬--------------┬--------------+
    |           |   vLLM       |  TensorRT-LLM|   Ollama     |
    +-----------+--------------+--------------+--------------+
    | 速度      | 快           | 最快         | 中等         |
    | 易用性    | 中等         | 复杂         | 最简单       |
    | 量化支持  | 好           | 好           | 好           |
    | KV Cache  | PagedAttn    | 优化         | 基础         |
    | 适用场景  | 生产服务     | 极致性能     | 本地开发     |
    +-----------+--------------+--------------+--------------┘

  部署策略：
    - 7B 模型 + 消费级 GPU（24GB）: INT4 量化 + vLLM
    - 7B 模型 + CPU only: GGUF 量化 + llama.cpp
    - 70B 模型 + 多 GPU: TP=4 + vLLM
    - 本地开发测试: Ollama（一行命令启动）
""")

# --- 代码验证：参数量计算 ---
print("\n[代码验证] 不同规模模型的参数量计算")
print("-" * 50)

def calc_params(vocab_size, d_model, n_layers, n_heads, n_kv_heads=None, ffn_mult=4, moe_experts=1, moe_topk=1):
    """计算 Transformer 模型的参数量"""
    if n_kv_heads is None:
        n_kv_heads = n_heads
    d_head = d_model // n_heads

    # Embedding
    emb_params = vocab_size * d_model

    # Per layer
    # QKV projections
    q_params = d_model * (n_heads * d_head)       # W_q
    k_params = d_model * (n_kv_heads * d_head)    # W_k
    v_params = d_model * (n_kv_heads * d_head)    # W_v
    o_params = (n_heads * d_head) * d_model        # W_o

    # FFN (or MoE)
    ffn_dim = int(ffn_mult * d_model * 2 / 3)  # SwiGLU style
    ffn_dim = ((ffn_dim + 63) // 64) * 64      # align to 64
    if moe_experts == 1:
        ffn_params = 3 * d_model * ffn_dim  # gate + up + down
    else:
        ffn_params = moe_experts * 3 * d_model * ffn_dim  # all experts
        ffn_params += d_model * moe_experts  # router

    # LayerNorm (RMSNorm)
    norm_params = 2 * d_model  # 2 per layer

    per_layer = q_params + k_params + v_params + o_params + ffn_params + norm_params
    total = emb_params + n_layers * per_layer + d_model  # + final norm

    return total

models = [
    ("MiniMind (26M)",  dict(vocab_size=6400, d_model=512, n_layers=8, n_heads=8, n_kv_heads=2)),
    ("LLaMA-2-7B",     dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=32)),
    ("LLaMA-2-70B",    dict(vocab_size=32000, d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8)),
    ("Mixtral 8x7B",   dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, moe_experts=8)),
]

for name, kwargs in models:
    params = calc_params(**kwargs)
    print(f"  {name}: {params/1e6:.1f}M params ({params/1e9:.2f}B)")


# ============================================================
# 第6部分：数学推导篇
# ============================================================

print("\n" + "=" * 70)
print("第6部分：数学推导篇 — 手推核心公式")
print("=" * 70)

print("""
============================================================
Q18: 手推 Self-Attention 的计算过程
============================================================

  输入: X in R^{nxd}  (n 个 token, d 维嵌入)

  步骤1: 线性投影
    Q = X * W_Q    K = X * W_K    V = X * W_V
    Q, K, V in R^{nxd_k}

  步骤2: 计算注意力分数
    S = Q * K^T / sqrt(d_k)    in R^{nxn}
    S[i][j] = q_i * k_j / sqrt(d_k)  (token i 对 token j 的关注程度)

  步骤3: Softmax 归一化
    A[i][j] = exp(S[i][j]) / sum_k exp(S[i][k])
    每行之和 = 1，表示 token i 对所有 token 的注意力分配

  步骤4: 加权求和
    Output = A * V    in R^{nxd_k}
    output_i = sum_j A[i][j] * v_j  (token i 的输出是所有 value 的加权平均)

  为什么除以 sqrt(d_k)？
    当 d_k 很大时，q*k 的方差也大，softmax 输出接近 one-hot
    除以 sqrt(d_k) 使方差归一化，softmax 输出更平滑，梯度更好

============================================================
Q19: 手推 DPO 损失函数的推导
============================================================

  第1步: RLHF 的目标是最大化奖励，同时不偏离参考模型太远
    max_pi E_{x,y~pi}[r(x,y)] - beta * KL(pi || pi_ref)

  第2步: 这个优化问题的闭式解（用拉格朗日对偶推导）
    pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)
    其中 Z(x) = sum_y pi_ref(y|x) * exp(r(x,y)/beta) 是配分函数

  第3步: 反解奖励函数（关键一步！）
    pi*(y|x) / pi_ref(y|x) = exp(r(x,y)/beta) / Z(x)
    log(pi*(y|x) / pi_ref(y|x)) = r(x,y)/beta - log Z(x)
    r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)

  第4步: 代入 Bradley-Terry 偏好模型
    P(y_w > y_l | x) = sigma(r(x,y_w) - r(x,y_l))

    r(x,y_w) - r(x,y_l)
    = beta * [log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))]
    （Z(x) 被消掉了！这就是 DPO 不需要奖励模型的原因）

  第5步: 最终 DPO 损失
    L_DPO = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))]

  直觉：让模型增大好回答的概率，减小坏回答的概率，同时不偏离参考模型太远

============================================================
Q20: 手推 LoRA 为什么低秩就够用
============================================================

  原始权重: W in R^{dxd}，参数量 d^2
  LoRA 分解: DeltaW = A * B，A in R^{dxr}, B in R^{rxd}，参数量 2dr

  参数压缩比: d^2 / (2dr) = d/(2r)
  当 d=4096, r=8 时: 压缩比 = 256x，只需 0.4% 的参数

  为什么低秩够用？
  1. 经验发现：微调时 DeltaW 的有效秩通常只有 8-64
     有效秩 = DeltaW 奇异值中"显著"的个数
     说明微调只改变了 8-64 个方向，其他方向几乎没变

  2. 理论解释：
     预训练模型已经在高维空间中找到了好的位置
     微调只需要在这个位置附近做小幅调整
     类比：装修时不需要拆墙重建，刷漆换窗帘就够了

  3. SVD 视角：
     DeltaW = U*S*V^T，大部分奇异值很小
     只保留前 r 个最大的奇异值: DeltaW ~= U_r*S_r*V_r^T
     这就是 LoRA 的 A*B（A=U_r*sqrtS_r, B=sqrtS_r*V_r^T）
""")

# --- 代码验证：LoRA 低秩有效性 ---
print("\n[代码验证] LoRA 低秩有效性：模拟微调权重变化")
print("-" * 50)

d = 512
W_pretrained = torch.randn(d, d)

# 模拟微调后的权重变化（低秩扰动）
true_rank = 8
A_true = torch.randn(d, true_rank) * 0.01
B_true = torch.randn(true_rank, d) * 0.01
delta_W = A_true @ B_true  # 真实的权重变化，秩=8

# SVD 分析
U, S, Vh = torch.linalg.svd(delta_W)
print(f"权重变化 DeltaW 的形状: {delta_W.shape}")
print(f"前10个奇异值: {S[:10].tolist()}")
print(f"奇异值衰减比 (第10个/第1个): {(S[9]/S[0]).item():.6f}")
print(f"有效秩（奇异值>最大值1%的个数）: {(S > S[0]*0.01).sum().item()}")
print("-> 大部分奇异值接近0，低秩近似就够了")


# ============================================================
# 第7部分：高频追问 & 易错点
# ============================================================

print("\n" + "=" * 70)
print("第7部分：高频追问 & 易错点")
print("=" * 70)

print("""
============================================================
Q21: 常见易错概念辨析
============================================================

1. GQA vs MQA vs MHA：
   MHA: 每个 head 有独立的 K,V（n_kv_heads = n_heads）
   GQA: K,V heads < Q heads，多个 Q head 共享一组 KV
   MQA: 所有 Q head 共享 1 组 KV（n_kv_heads = 1，GQA 的极端情况）

2. Temperature vs Top-K vs Top-P：
   Temperature: 控制分布的"尖锐度"（T(down)->更确定，T(up)->更随机）
   Top-K: 只从概率最高的 K 个 token 中采样
   Top-P: 只从累积概率达到 P 的最小 token 集合中采样
   三者可以组合使用：先 Temperature 缩放，再 Top-K 截断，再 Top-P 过滤

3. Pre-training Loss vs SFT Loss：
   预训练 loss 衡量"预测下一个 token 的能力"
   SFT loss 衡量"按指令回答的能力"
   预训练 loss 低 != 回答问题好（需要 SFT 对齐）

4. Embedding vs Hidden State：
   Embedding: 词到向量的映射（查表操作，nn.Embedding）
   Hidden State: 经过 Transformer 层处理后的表示
   Embedding 是"初始理解"，Hidden State 是"深层理解"

5. Cross-Entropy vs KL Divergence：
   Cross-Entropy: H(p, q) = -sum p(x) log q(x)  （预测 q 和真实 p 的差距）
   KL Divergence: KL(p||q) = H(p,q) - H(p)     （多了一个熵项）
   当 p 是 one-hot 时（如标签），CE = KL（因为 H(p)=0）

============================================================
Q22: 2024-2025 LLM 新特性速览
============================================================

1. DeepSeek-V3 (2024.12):
   - MoE + MLA（Multi-head Latent Attention，KV Cache 压缩新方案）
   - FP8 混合精度训练（首次在超大规模验证 FP8 训练可行）
   - GRPO 强化学习（省掉价值模型）
   - 671B 参数，37B 激活（每 token 只用 5.5% 的参数）

2. MLA（Multi-head Latent Attention）:
   - 核心思想：把 KV 压缩到低维潜在空间
   - KV Cache 不存原始 k,v，存压缩后的 c_kv（维度远小于 d_model）
   - 推理时从 c_kv 解压回 k,v
   - 比 GQA 更进一步：GQA 减少头数，MLA 减少维度

3. LLaMA 3 (2024.4):
   - 8B/70B/405B 三个规模
   - 128K 上下文（通过长上下文扩展训练）
   - GQA（8B 用 GQA，70B 也用 GQA）
   - 训练数据 15T tokens

4. Mamba-2 (2024.5):
   - SSM + Attention 统一框架（SSD, Structured State Space Duality）
   - 比 Mamba-1 快 2-8x
   - 证明 SSM 和 Attention 在数学上是特例关系

5. 量化新方法:
   - GPTQ: 基于近似二阶信息的训练后量化
   - AWQ: 基于激活感知权重的量化（保护重要权重）
   - GGUF: llama.cpp 的量化格式，支持 CPU 推理
   - FP8 训练: DeepSeek-V3 首次大规模验证

6. 推理框架:
   - vLLM: PagedAttention，高吞吐推理
   - SGLang: 编程式 LLM 调用，RadixAttention
   - TensorRT-LLM: NVIDIA 官方，极致性能

============================================================
Q23: 面试高频手撕题
============================================================

1. 手写 Self-Attention（最常考）
2. 手写 Multi-Head Attention
3. 手写 RoPE 位置编码
4. 手写 SwiGLU FFN
5. 手写 RMSNorm
6. 手写 GQA
7. 手写 KV Cache + 自回归生成
8. 手写 DPO Loss
9. 手写 LoRA 线性层
10. 手写简单的 MoE 层
""")

# --- 手撕代码演示 ---
print("\n[手撕代码] 面试必背：Self-Attention")
print("-" * 50)

def self_attention(X, W_q, W_k, W_v):
    """最简洁的 Self-Attention 实现"""
    Q = X @ W_q                    # (n, d) @ (d, d_k) -> (n, d_k)
    K = X @ W_k
    V = X @ W_v
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)   # (n, n)
    attn = F.softmax(scores, dim=-1)     # (n, n)
    return attn @ V                       # (n, d_k)

n, d, d_k = 4, 8, 8
X = torch.randn(n, d)
W_q = torch.randn(d, d_k)
W_k = torch.randn(d, d_k)
W_v = torch.randn(d, d_k)
output = self_attention(X, W_q, W_k, W_v)
print(f"输入: {X.shape} -> 输出: {output.shape}")
print(f"输出前2个token: {output[:2, :4].detach().tolist()}")

print("\n[手撕代码] 面试必背：RoPE")
print("-" * 50)

def apply_rope(x, seq_len, dim):
    """RoPE 旋转位置编码"""
    # x: (batch, seq_len, n_heads, d_head)
    positions = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # (d_head/2,)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, d_head/2)
    cos_val = torch.cos(angles)  # (seq_len, d_head/2)
    sin_val = torch.sin(angles)  # (seq_len, d_head/2)
    # Reshape for broadcasting: (1, seq_len, 1, d_head/2)
    cos_val = cos_val.unsqueeze(0).unsqueeze(2)
    sin_val = sin_val.unsqueeze(0).unsqueeze(2)

    # 将 x 的相邻维度配对旋转
    x1, x2 = x[..., ::2], x[..., 1::2]  # (1, seq_len, n_heads, d_head/2)
    out_x1 = x1 * cos_val - x2 * sin_val
    out_x2 = x1 * sin_val + x2 * cos_val
    return torch.stack([out_x1, out_x2], dim=-1).flatten(-2)

x_test = torch.randn(1, 8, 2, 16)  # (batch=1, seq=8, heads=2, d_head=16)
x_rope = apply_rope(x_test, 8, 16)
print(f"RoPE 输入: {x_test.shape} -> 输出: {x_rope.shape}")
print("-> 相邻维度配对旋转，编码了位置信息")

print("\n[手撕代码] 面试必背：DPO Loss")
print("-" * 50)

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO 损失函数"""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return loss.mean()

# 模拟数据
p_chosen = torch.tensor([-1.2, -0.8, -1.5])
p_rejected = torch.tensor([-2.1, -1.9, -2.3])
r_chosen = torch.tensor([-1.0, -0.7, -1.3])
r_rejected = torch.tensor([-1.8, -1.6, -2.0])

loss = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected)
print(f"DPO Loss: {loss.item():.4f}")
print("-> 让 chosen 的 reward 比 rejected 高（logsigmoid 保证单调性）")

print("\n[手撕代码] 面试必背：LoRA Linear")
print("-" * 50)

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

lora_layer = LoRALinear(512, 512, rank=8)
base_params = sum(p.numel() for p in lora_layer.linear.parameters())
lora_params = sum(p.numel() for p in [lora_layer.lora_A, lora_layer.lora_B])
print(f"原始参数: {base_params:,}")
print(f"LoRA参数: {lora_params:,} ({lora_params/base_params*100:.1f}%)")
print(f"压缩比: {base_params/lora_params:.0f}x")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("面试准备路线图")
print("=" * 70)

print("""
+-------------------------------------------------------------+
| Level 1 — 必须掌握（初级岗）                                |
|  * Self-Attention 原理和计算                                |
|  * Transformer 整体架构                                     |
|  * 预训练/SFT/RLHF 三阶段                                  |
|  * KV Cache 原理                                           |
|  * 量化基础概念                                             |
|  * 手写 Self-Attention                                     |
+-------------------------------------------------------------+
| Level 2 — 深入理解（中级岗）                                |
|  * RoPE 位置编码                                           |
|  * GQA/MQA/MHA 区别                                       |
|  * DPO vs RLHF 推导                                       |
|  * LoRA 原理和实现                                         |
|  * MoE 架构和负载均衡                                      |
|  * 分布式训练策略                                           |
|  * 手写 RoPE / DPO Loss / LoRA                             |
+-------------------------------------------------------------+
| Level 3 — 前沿掌握（高级岗）                                |
|  * Mamba/SSM 原理                                          |
|  * MLA (Multi-head Latent Attention)                       |
|  * GRPO 训练流程                                           |
|  * 投机解码                                                 |
|  * 长上下文扩展技术                                         |
|  * 多模态架构                                               |
|  * FP8 训练                                                 |
|  * 手写 MoE / KV Cache + 生成                               |
+-------------------------------------------------------------┘

面试技巧：
  1. 先说直觉类比，再说数学公式（展示理解深度）
  2. 主动说"我还可以展开讲XXX"（引导面试官问你擅长的）
  3. 不确定就说"据我了解...，但细节可能需要确认"（诚实比瞎编好）
  4. 代码题先写框架再填细节（展示工程能力）
  5. 结合实际项目经验（"我在 MiniMind 中实现过..."）

恭喜你完成了全部27课的学习！
""")
