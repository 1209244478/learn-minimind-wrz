"""
进阶课：MiniMind 的高级特性
============================

基础课中我们学习了 GPT 的核心组件。现在探索 MiniMind 的高级特性：

  1. MoE (Mixture of Experts): 混合专家，让不同专家处理不同内容
  2. TTT (Test-Time Training): 推理时训练，模型在推理中自我进化
  3. MTP (Multi-Token Prediction): 多Token预测，一次预测多个未来词
  4. KV Cache: 加速生成的关键优化
  5. MSA (MiniMax Sparse Attention): 稀疏注意力，处理超长文本

运行: python lessons/lesson11_advanced.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. MoE — 混合专家
# ============================================================

print("=" * 60)
print("[1] MoE — 混合专家：让不同专家处理不同内容")
print("=" * 60)

class SimpleExpert(nn.Module):
    """单个专家：一个标准的 SwiGLU FFN"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleMoE(nn.Module):
    """简化版 MoE：门控路由 + 多个专家

    与 MiniMind 原始项目的 MOEFeedForward 一致，包含：
    - norm_topk_prob: 归一化 Top-K 权重（确保权重和为1）
    - aux_loss: 负载均衡辅助损失（防止所有 token 被路由到同一个专家）
    """

    def __init__(self, hidden_size, intermediate_size, num_experts=4, top_k=2,
                 norm_topk_prob=True, router_aux_loss_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        self.experts = nn.ModuleList([
            SimpleExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)

        gate_scores = F.softmax(self.gate(x_flat), dim=-1)

        topk_weight, topk_idx = torch.topk(gate_scores, k=self.top_k, dim=-1, sorted=False)

        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * gate_scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = gate_scores.new_zeros(1).squeeze()

        return y.reshape(bsz, seq_len, hidden)


# 实验
hidden_size = 64
intermediate_size = 128

moe = SimpleMoE(hidden_size, intermediate_size, num_experts=4, top_k=2)
ffn = SimpleExpert(hidden_size, intermediate_size)

x = torch.randn(2, 8, hidden_size)
out_moe = moe(x)
out_ffn = ffn(x)

moe_params = sum(p.numel() for p in moe.parameters())
ffn_params = sum(p.numel() for p in ffn.parameters())

print(f"标准 FFN 参数量: {ffn_params:,}")
print(f"MoE (4专家, Top-2) 参数量: {moe_params:,} ({moe_params/ffn_params:.1f}x)")
print(f"但每次推理只用 Top-2 个专家，实际计算量 ≈ {2/4*100:.0f}%")
print(f"\nMoE 的核心思想:")
print(f"  - 参数量大 → 知识丰富")
print(f"  - 每次只用部分专家 → 计算量可控")
print(f"  - 门控网络自动选择最合适的专家")
print(f"  - 不同类型的 token 被路由到不同的专家")


# ============================================================
# 2. TTT — 推理时训练
# ============================================================

print("\n" + "=" * 60)
print("[2] TTT — 推理时训练：模型在推理中自我进化")
print("=" * 60)

print("""
TTT (In-Place Test-Time Training) 的核心思想：

  普通模型：训练后参数固定，推理时不更新
  TTT 模型：推理时用当前输入微调参数，让模型适应当前上下文

  具体做法：
  1. 在 FFN 的 down_proj 上做 mini-batch SGD
  2. 用 next-token prediction 作为自监督信号
  3. 每次推理更新几步，将上下文信息压缩进权重

  类比：
    普通模型 = 考试时只能用已有的知识
    TTT 模型 = 考试时可以翻书学习（但只学当前题目相关的）

  MiniMind 的 TTT 实现：
  - 只在 FFN 的 down_proj 上更新
  - 用轻量线性头构建 next-token prediction 信号
  - 每个 chunk（如512个token）重置权重
  - 学习率很小（1e-4），避免过度修改
""")

# 简化版 TTT 演示
class SimpleTTTLayer(nn.Module):
    """简化版 TTT：推理时微调 down_proj"""

    def __init__(self, hidden_size, intermediate_size, ttt_lr=1e-4):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.ttt_lr = ttt_lr
        self.ttt_predictor = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, use_ttt=False):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        if use_ttt and not self.training:
            # TTT: 在推理时微调 down_proj
            W = self.down_proj.weight.clone()
            for t in range(x.shape[1] - 1):
                # 当前位置的输出
                out = F.linear(hidden[:, t], W)
                # 自监督：用输出预测下一个位置的输入
                pred = self.ttt_predictor(out)
                target = x[:, t + 1]
                loss = F.mse_loss(pred, target.detach())
                # 计算梯度并更新
                grad = torch.autograd.grad(loss, W, retain_graph=False)[0]
                W = W - self.ttt_lr * grad

            # 用更新后的权重做最终输出
            output = F.linear(hidden, W)
        else:
            output = self.down_proj(hidden)

        return output

ttt_layer = SimpleTTTLayer(64, 128)
x = torch.randn(1, 8, 64)

out_normal = ttt_layer(x, use_ttt=False)
out_ttt = ttt_layer(x, use_ttt=True)

print(f"普通推理输出 std: {out_normal.std():.4f}")
print(f"TTT推理输出 std:  {out_ttt.std():.4f}")
print(f"TTT 让模型在推理时适应输入，输出可能不同")


# ============================================================
# 3. MTP — 多Token预测
# ============================================================

print("\n" + "=" * 60)
print("[3] MTP — 多Token预测：一次预测多个未来词")
print("=" * 60)

print("""
标准语言模型：每次只预测下一个词
MTP (Multi-Token Prediction)：同时预测未来 N 个词

为什么需要 MTP？
  1. 训练信号更丰富：每个位置不只学1个目标，学N个
  2. 更好的规划能力：模型必须"想清楚"未来几步
  3. 推理时可并行生成多个token，加速

实现方式：
  在 lm_head 后追加 N 个预测头：
    head_0: 预测 t+1 位置的词（标准）
    head_1: 预测 t+2 位置的词
    head_2: 预测 t+3 位置的词
    ...

  每个预测头有自己的 Norm + Linear + Residual
""")

class SimpleMTPHead(nn.Module):
    """一个 MTP 预测头"""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x, base_logits):
        # 残差连接到基础 logits
        projected = self.proj(self.norm(x))
        return base_logits + self.lm_head(projected)


# 演示
vocab_size = 200
hidden_size = 64
num_mtp_heads = 3

base_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
mtp_heads = nn.ModuleList([
    SimpleMTPHead(hidden_size, vocab_size)
    for _ in range(num_mtp_heads)
])

x = torch.randn(1, 8, hidden_size)
base_logits = base_lm_head(x)

print(f"基础预测头: 预测位置 t+1")
for i, head in enumerate(mtp_heads):
    logits = head(x, base_logits)
    print(f"MTP头 {i+1}: 预测位置 t+{i+2}, logits形状={logits.shape}")

print(f"\n训练时：4个预测头的损失加权求和")
print(f"  total_loss = loss_head0 + 0.1 * (loss_head1 + loss_head2 + loss_head3)")
print(f"  MTP辅助损失权重通常较小（0.1），避免干扰主预测")


# ============================================================
# 4. KV Cache — 加速生成的关键优化
# ============================================================

print("\n" + "=" * 60)
print("[4] KV Cache — 避免重复计算")
print("=" * 60)

class SimpleKVCache:
    """简化版 KV Cache"""

    def __init__(self, num_layers, max_len, num_kv_heads, head_dim, device='cpu'):
        self.max_len = max_len
        self.k_cache = [torch.zeros(1, max_len, num_kv_heads, head_dim, device=device) for _ in range(num_layers)]
        self.v_cache = [torch.zeros(1, max_len, num_kv_heads, head_dim, device=device) for _ in range(num_layers)]
        self.cur_len = [0] * num_layers

    def update(self, layer_idx, new_k, new_v):
        cur = self.cur_len[layer_idx]
        new_len = cur + new_k.shape[1]
        self.k_cache[layer_idx][:, cur:new_len].copy_(new_k)
        self.v_cache[layer_idx][:, cur:new_len].copy_(new_v)
        self.cur_len[layer_idx] = new_len
        return self.k_cache[layer_idx][:, :new_len], self.v_cache[layer_idx][:, :new_len]


# 对比有无 KV Cache 的计算量
import time

class SimpleModelWithKVCache(nn.Module):
    """支持 KV Cache 的简化模型"""

    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*4,
            batch_first=True, dropout=0.0
        ) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

model_cache = SimpleModelWithKVCache(200, 64)
model_cache.eval()

# 朴素生成
input_ids = torch.randint(0, 200, (1, 4))
start = time.time()
generated = input_ids.clone()
with torch.no_grad():
    for _ in range(20):
        logits = model_cache(generated)
        next_tok = logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_tok], dim=1)
time_no_cache = (time.time() - start) * 1000

# 统计计算量
total_tokens_processed = sum(4 + i for i in range(20))
print(f"朴素生成 20 个 token:")
print(f"  总共处理了 {total_tokens_processed} 个 token 的前向传播")
print(f"  耗时: {time_no_cache:.1f} ms")
print(f"\nKV Cache 生成:")
print(f"  第1步: 处理 4 个 token（prefill）")
print(f"  第2-20步: 每步只处理 1 个 token（decode）")
print(f"  总共处理: 4 + 19 = 23 个 token（vs {total_tokens_processed}）")
print(f"  节省: {(1 - 23/total_tokens_processed)*100:.0f}% 的计算量")


# ============================================================
# 5. MSA — 稀疏注意力
# ============================================================

print("\n" + "=" * 60)
print("[5] MSA — 稀疏注意力：处理超长文本")
print("=" * 60)

print("""
标准 Attention 的问题：
  计算量 = O(N^2)，N 是序列长度
  N=1024 → 1M 次运算
  N=8192 → 67M 次运算 → 太慢！

MSA (MiniMax Sparse Attention) 的解决方案：
  不是每个 token 都关注所有其他 token，而是只关注"重要"的 token

  两阶段设计：
  ┌─────────────────────────────────────┐
  │  Stage 1: Index Branch（索引分支）    │
  │  - 用低维投影计算每个块的相关性分数    │
  │  - 选出 Top-K 个最重要的块           │
  │  - 计算量小，快速筛选                │
  └─────────────────────────────────────┘
                 ↓ 选出的块索引
  ┌─────────────────────────────────────┐
  │  Stage 2: Sparse Branch（稀疏分支）  │
  │  - 只对选中的块做完整注意力计算       │
  │  - 跳过不相关的块，节省计算          │
  │  - GQA 组共享索引，进一步减少开销     │
  └─────────────────────────────────────┘

  短序列（< fallback_len）: 自动退化为标准注意力
  长序列: 只计算 top-k 比例的块，复杂度 O(N * top_k * block_size)
""")

# 演示块级选择
seq_len = 128
block_size = 16
n_blocks = seq_len // block_size
topk_ratio = 0.25
topk = max(1, int(n_blocks * topk_ratio))

print(f"序列长度: {seq_len}, 块大小: {block_size}")
print(f"总块数: {n_blocks}, Top-K: {topk} (比例={topk_ratio})")

# 模拟索引分支的块选择
torch.manual_seed(42)
block_scores = torch.randn(1, seq_len, 1, n_blocks)
_, topk_indices = torch.topk(block_scores, k=topk, dim=-1)

print(f"\n每个位置只关注 {topk}/{n_blocks} 个块 = {topk/n_blocks*100:.0f}%")
print(f"标准注意力: 每个位置关注 {seq_len} 个 token")
print(f"MSA注意力: 每个位置关注 {topk * block_size} 个 token (约)")
print(f"计算量节省: ~{(1 - topk/n_blocks)*100:.0f}%")

print(f"\nMSA 的关键设计:")
print(f"  1. 块级选择（block_size=64）：粗粒度筛选，减少索引开销")
print(f"  2. GQA 组共享索引：同一组的 Q 头共享选择结果")
print(f"  3. Fallback 机制：短序列自动退化为标准注意力")
print(f"  4. 因果掩码：确保只能看过去的块")


# ============================================================
# 进阶课程总结
# ============================================================

print("\n" + "=" * 60)
print("进阶课程总结")
print("=" * 60)
print("""
MiniMind 的5大高级特性：

1. MoE (混合专家)
   - 参数量大但计算量可控
   - 门控网络自动路由
   - 适合扩大模型规模而不增加推理成本

2. TTT (推理时训练)
   - 推理时微调参数，适应上下文
   - 自监督信号：next-token prediction
   - 让模型在推理中"学习"

3. MTP (多Token预测)
   - 同时预测未来N个词
   - 训练信号更丰富，规划能力更强
   - 推理时可并行生成

4. KV Cache
   - 缓存已计算的KV，避免重复计算
   - 生成速度从O(N^2)降到O(N)
   - 预分配缓冲区，避免动态拼接

5. MSA (稀疏注意力)
   - 两阶段：索引分支筛选 + 稀疏分支计算
   - 长序列计算量大幅降低
   - 短序列自动退化为标准注意力

这些特性让 MiniMind 在小模型规模下实现了强大的能力！
恭喜你完成了 MiniMind 的全部学习课程！
""")


# ============================================================
# 深入理解：五大高级特性对比
# ============================================================
print("\n" + "=" * 60)
print("深入理解：五大高级特性对比")
print("=" * 60)

print("""
【类比：LLM 的'超能力'】
─────────────────────
  MoE    = 多个专家会诊, 哪个对口用哪个
  TTT    = 看一遍就记住, 推理时还会改笔记
  MTP    = 走一步看三步, 提前规划
  KV     = 记下之前看过的书页, 不用翻回去
  MSA    = 找重点, 不重要的内容快速略过


【五大特性解决什么问题？】
──────────────────────

  MoE (Mixture of Experts):
    问题: 模型变大 = 计算量变大
    解决: 每次只激活少数专家, 容量大但计算少
    类比: 大医院, 病人来了挂号分诊, 不用所有医生都看
    关键: 门控网络 (router) 决定哪个专家处理

  TTT (Test-Time Training):
    问题: 上下文信息难以融入模型
    解决: 推理时还在"学习" (微调 down_proj)
    类比: 边看书边做笔记, 越看越懂
    关键: 不增加参数, 只在推理时临时调整

  MTP (Multi-Token Prediction):
    问题: 一次只预测一个词, 训练慢
    解决: 一次看多步, 同时预测下 k 个词
    类比: 学下棋, 走一步看三步
    关键: 多输出头 + 共享主干

  KV Cache:
    问题: 生成时重复计算 K, V
    解决: 把 K, V 存起来, 一次算好
    类比: 考试时把参考书摊开, 不用反复合上翻开
    关键: 预分配 buffer, 避免动态拼接

  MSA (MiniMax Sparse Attention):
    问题: 注意力计算量随序列长度平方增长
    解决: 只对重要位置算完整注意力
    类比: 做阅读理解时, 重点段落精读, 其他略读
    关键: Index 选 top-k 块, 稀疏计算


【图示：KV Cache vs MSA 区别】
─────────────────────────

  KV Cache (加速生成, 不改变计算):
    
    无缓存:
      步骤1: [token1, token2, token3]           → 算 K1, V1, K2, V2, K3, V3
      步骤2: [token1, token2, token3, t1_new]   → 算 K1, V1, K2, V2, K3, V3, K4, V4  ← 重复!
      步骤3: [token1, ..., t1_new, t2_new]      → 算 K1, V1, ..., K4, V4, K5, V5   ← 重复!
    
    有缓存:
      步骤1: [token1, token2, token3]           → 算 K1, V1, K2, V2, K3, V3
      步骤2: [t1_new] (只算新的)                → 算 K4, V4, 复用前面的
      步骤3: [t2_new]                           → 算 K5, V5, 复用前面的

  MSA (改变注意力计算方式, 降低复杂度):
    
    标准注意力:
      Q K^T → 所有位置两两算相似度
      O(N²) 计算量, N=4096 时 = 16M 元素
    
    稀疏注意力:
      Q K^T → 只对 top-k 重要位置算完整
      O(N * k) 计算量, k=512 时 = 2M 元素
      加速 8x!

  → KV Cache 解决"重复计算", 适用所有 Transformer
  → MSA 解决"全连接注意力浪费", 适用长序列


【何时启用这些特性？】
──────────────────

  模型规模:
    小 (< 1B):  不需要 MoE, KV Cache 够用
    中 (1B-7B): 加 KV Cache, MTP 训练, MSA
    大 (> 7B):  MoE, TTT, MSA 全开

  任务:
    长文本 (> 4K): 必开 MSA, KV Cache
    多任务: MoE 让不同任务用不同专家
    推理增强: TTT 让模型在线学习

  部署环境:
    显存紧: KV Cache 量化, MoE 卸载
    速度紧: MSA, KV Cache 都要
    精度紧: TTT, MTP

  MiniMind 现状:
    实现了五大特性, 可根据配置开关
    用户可根据硬件和任务选择


【综合实验：观察各特性效果】
─────────────────────────
""")

# 简单实验: 对比有无 KV Cache
import time

print("实验 1: KV Cache 加速效果")
print("-" * 40)

seq_lens_to_test = [64, 128, 256]
print(f"{'Seq Len':<12} {'无 Cache (ms)':<16} {'有 Cache (ms)':<16} {'加速比':<10}")
print("-" * 60)

# 模拟简单的 Q @ K^T
for seq_len in seq_lens_to_test:
    Q = torch.randn(1, 8, seq_len, 64)
    K = torch.randn(1, 8, seq_len, 64)
    V = torch.randn(1, 8, seq_len, 64)

    # 无缓存
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(5):
        attn = torch.matmul(Q, K.transpose(-2, -1))
        out = torch.matmul(attn, V)
    t1 = time.time()
    no_cache_time = (t1 - t0) * 100

    # 有缓存: 只算新 token
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(5):
        q_new = Q[:, :, -1:]
        attn = torch.matmul(q_new, K.transpose(-2, -1))
        out = torch.matmul(attn, V)
    t1 = time.time()
    cache_time = (t1 - t0) * 100

    speedup = no_cache_time / max(cache_time, 1e-6)
    print(f"{seq_len:<12} {no_cache_time:<16.2f} {cache_time:<16.2f} {speedup:<10.2f}x")

print("\n实验 2: 稀疏注意力 vs 全注意力 (理论)")
print("-" * 40)

for N in [512, 1024, 2048, 4096, 8192]:
    full = N * N
    sparse = N * 256  # top-256
    saving = (1 - sparse / full) * 100
    print(f"  N={N:5d}: 全注意力 {full**1:>14,} → 稀疏 {sparse:>9,} (节省 {saving:.1f}%)")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】MoE 路由
  8 个专家, top-2 路由, 一次推理激活几个专家? 节省多少计算?

【练习2】TTT vs 微调
  TTT 和传统微调的本质区别是什么?

【练习3】MTP 优势
  MTP 比单 token 预测多了多少损失? k=4 时

【练习4】KV Cache 显存
  生成 4096 tokens, batch=1, 32 层, 8 KV 头, head_dim=128
  KV Cache 占多少显存? (FP16)

【练习5】MSA 退化
  MSA 在短序列 (N=128) 时为什么退化为标准注意力?

【练习6】特性组合
  训练时哪些特性影响 loss? 推理时哪些影响延迟?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  8 个专家, top-2 路由:")
print("    激活: 2 个专家")
print("    节省: (8-2)/8 = 75% 计算")
print()
print("  实际计算量 (相对):")
print("    全专家激活: 8x")
print("    top-2 激活: 2x")
print("    节省: 6x (75% off)")
print()
print("  进一步考虑:")
print("    - 门控网络本身有计算量 (8 logits → softmax → top-2)")
print("    - 但相对专家计算量很小")
print("    - 净节省约 70%")

# 练习2
print("\n【练习2 答案】")
print("  本质区别:")
print()
print("  传统微调:")
print("    - 大量数据 + 大量算力 + 训练后更新所有参数")
print("    - 微调后, 模型固定, 不再变化")
print("    - 类比: 寒暑假集中培训")
print()
print("  TTT:")
print("    - 推理时针对当前输入, 在线微调少量参数")
print("    - 只改 down_proj (1 个矩阵)")
print("    - 不持久化, 输入变就忘")
print("    - 类比: 边看病边查资料")
print()
print("  关键差异:")
print("    - 微调: 离线, 持久, 全参数")
print("    - TTT: 在线, 临时, 局部参数")
print()
print("  实际应用:")
print("    TTT 适合: 个性化对话, 长文档问答")
print("    微调适合: 通用能力提升, 风格定制")

# 练习3
print("\n【练习3 答案】")
print("  MTP k=4, 比单 token 多了 4 倍预测损失")
print()
print("  损失组成:")
print("    L_total = L_1 + L_2 + L_3 + L_4")
print("    L_i = 预测第 i 个未来 token 的交叉熵")
print()
print("  实际:")
print("    L_1 = 直接的 next token 损失")
print("    L_2, L_3, L_4 = 间接监督")
print("    → 主损失还是 L_1, 其他是辅助")
print()
print("  训练中:")
print("    L_total = L_1 + 0.5 * (L_2 + L_3 + L_4)  (加权)")
print("    → 辅助损失权重要低, 避免喧宾夺主")
print()
print("  推理时:")
print("    MTP 头不参与, 推理速度和单 token 一致")

# 练习4
print("\n【练习4 答案】")
seq_len = 4096
batch = 1
num_layers = 32
num_kv_heads = 8
head_dim = 128
fp16_bytes = 2

total_elements = 2 * batch * num_layers * seq_len * num_kv_heads * head_dim
total_bytes = total_elements * fp16_bytes
total_mb = total_bytes / (1024 * 1024)
total_gb = total_mb / 1024

print(f"  配置: bs={batch}, layers={num_layers}, seq={seq_len}")
print(f"  num_kv_heads={num_kv_heads}, head_dim={head_dim}")
print(f"  K, V 各 {batch} × {num_layers} × {seq_len} × {num_kv_heads} × {head_dim}")
print(f"  K+V 总元素: {total_elements:,}")
print(f"  字节: {total_bytes:,} = {total_mb:.1f} MB = {total_gb:.2f} GB")
print()
print(f"  → 单序列 4K 上下文, KV Cache 已经 0.5 GB")
print(f"  → 这就是为什么长上下文需要 KV Cache 量化")
print(f"  → PagedAttention (vLLM) 通过分页进一步优化")

# 练习5
print("\n【练习5 答案】")
print("  短序列退化的原因:")
print()
print("  1. 稀疏收益太小:")
print("    N=128, top-k=64 → 节省 50%")
print("    N=4096, top-k=64 → 节省 98%")
print("    短序列时, 全连接的常数小, 稀疏反而引入开销")
print()
print("  2. 索引分支开销固定:")
print("    索引分支要算 N 个 block 分数")
print("    N 小, 这部分开销占比反而大")
print()
print("  3. 退化策略:")
print("    if N < threshold:  # 通常 threshold=512-1024")
print("        use standard attention")
print("    else:")
print("        use sparse attention")
print()
print("  MiniMind 实现:")
print("    有 fallback 机制, 短序列自动用标准注意力")
print("    避免稀疏带来的额外开销")

# 练习6
print("\n【练习6 答案】")
print("  训练时影响 loss:")
print("    - MTP: 增加辅助损失, 直接影响")
print("    - MoE: 影响路由分布, 间接影响")
print("    - TTT: 仅推理时使用")
print("    - KV Cache: 训练不用, 加速用")
print("    - MSA: 训练和推理都可用")
print()
print("  推理时影响延迟:")
print("    - KV Cache: 极大降低延迟 (N 倍加速)")
print("    - MSA: 降低延迟 (稀疏计算)")
print("    - MoE: 增加延迟 (路由开销) 但能跑更大模型")
print("    - TTT: 增加延迟 (推理时训练)")
print("    - MTP: 推理不参与, 不影响")
print()
print("  实际部署选择:")
print("    速度优先: KV Cache + MSA")
print("    容量优先: MoE + KV Cache")
print("    质量优先: TTT + MTP + MSA")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. MoE/TTT/MTP/KV Cache/MSA 是 MiniMind 的五大高级特性
2. 它们解决不同问题：效率/质量/速度/容量
3. 训练时影响 loss：MTP > MoE > MSA > TTT > KV Cache
4. 推理时影响延迟：KV Cache 最大, MSA 次之
5. 可根据硬件和任务灵活选择和组合

MiniMind 通过这些特性, 在小模型规模下实现强大能力！
""")
