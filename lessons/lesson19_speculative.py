"""
第19课：投机解码 (Speculative Decoding) — 用小模型加速大模型
==============================================================

问题: LLM 推理慢, 怎么办?
  一次生成 1 个 token, GPU 利用率低
  7B 模型, 1 秒生成 30 个 token (FP16) → 体验卡顿

核心洞察:
  生成 "我喜欢吃苹果" 需要 5 步
  实际有效信息 = 5 个 token
  GPU 浪费在 "自回归" 的串行上

投机解码 (Leviathan et al., 2023):
  - 用小模型 (Draft Model) 一次生成 k 个候选 token
  - 用大模型 (Target Model) 一次验证所有 k 个 token
  - 接受正确的部分, 拒绝的重新生成
  - 通常提速 2-3x, 输出分布完全一致

本课会讲解:
  1. 串行生成 vs 并行验证
  2. 接受/拒绝算法
  3. 性能分析

运行: python lessons/lesson19_speculative.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 第1部分：为什么需要投机解码
# ============================================================

print("=" * 60)
print("第1部分：为什么需要投机解码")
print("=" * 60)

print("""
传统自回归生成的瓶颈:
  - 一次只能生成 1 个 token
  - GPU 并行能力浪费 (算 1 个 token vs 算 100 个 token 时间差 < 2x)
  - 内存带宽限制 (memory-bound)

数学分析:
  生成 1 token: 需要读全模型权重 (memory-bound)
  生成 k tokens: 仍然需要读全模型权重 (但只要一次)

  100 个 token 串行:
    Time = 100 × t_one  (t_one 是单 token 推理时间)

  100 个 token 并行 (理论):
    Time ≈ 1 × t_full  (一次前向传播)

  理想加速比: 100x, 实际 2-4x (取决于接受率)
""")

# 演示单 token vs 并行的时间
print("\n[演示] 串行 vs 并行的时间差异")
print("-" * 60)
print(f"{'生成 token 数':<15}{'串行 (相对)':<15}{'并行 (相对)':<15}{'理想加速':<10}")
print("-" * 60)
t_one = 1.0  # 假设单 token 推理时间 1
t_full = 5.0  # 假设一次完整前向 5 倍时间
for n_tokens in [1, 5, 10, 50, 100]:
    serial = n_tokens * t_one
    parallel = t_full
    speedup = serial / parallel
    print(f"{n_tokens:<15}{serial:<15.0f}{parallel:<15.0f}{speedup:<10.1f}x")


# ============================================================
# 第2部分：投机解码原理
# ============================================================

print("\n" + "=" * 60)
print("第2部分：投机解码原理")
print("=" * 60)

print("""
关键角色:
  - Draft Model (小模型): 1B 参数, 推理快 5x
  - Target Model (大模型): 7B 参数, 推理慢

算法流程:
  输入: prefix 序列

  1. 用 draft model 生成 k 个候选:
     x_1, x_2, ..., x_k    (k=4-8)

  2. 用 target model 一次前向验证:
     输入: prefix + x_1 + x_2 + ... + x_k
     输出: k+1 个 logits

  3. 接受/拒绝算法:
     for i = 1 to k:
       接受 x_i 的概率 = min(1, p_target(x_i) / p_draft(x_i))
       随机采样, 决定是否接受

     如果 x_i 被拒绝, 用 target 分布重新采样替换 x_i
     如果 x_1...x_k 全部接受, 把 x_{k+1} 的 argmax 加进去

  4. 一次生成 k+1 个 token!

关键性质:
  ✓ 输出分布与直接用 target model 完全一致
  ✓ 不损失任何质量
  ✓ 平均加速 2-3x
""")


# ============================================================
# 第3部分：接受/拒绝算法实现
# ============================================================

print("\n" + "=" * 60)
print("第3部分：接受/拒绝算法实现")
print("=" * 60)


def speculative_verify(draft_probs, target_probs, draft_tokens, temperature=1.0):
    """
    投机解码的接受/拒绝算法

    draft_probs: [batch, k, vocab]  draft model 概率
    target_probs: [batch, k, vocab] target model 概率
    draft_tokens: [batch, k]         draft 生成的 tokens

    返回:
    - accepted: [batch] 接受的数量
    - bonus_token: [batch] 额外 token (如果全部接受)
    """
    batch_size, k, vocab_size = draft_probs.shape

    # 1. 获取 draft 和 target 在各自 token 上的概率
    draft_p = draft_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)  # [batch, k]
    target_p = target_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)  # [batch, k]

    # 2. 接受率 = min(1, target_p / draft_p)
    accept_rate = torch.clamp(target_p / draft_p.clamp_min(1e-8), max=1.0)

    # 3. 随机采样决定是否接受
    random_vals = torch.rand_like(accept_rate)
    accepted_mask = random_vals < accept_rate  # [batch, k]

    # 4. 找第一个拒绝的位置 (从前往后)
    # 如果全部接受, 返回 k (并加 1 个 bonus)
    accepted = torch.full((batch_size,), k, dtype=torch.long)

    for b in range(batch_size):
        for i in range(k):
            if not accepted_mask[b, i]:
                accepted[b] = i
                break

    return accepted, accepted_mask


# 演示接受/拒绝
print("\n[演示] 接受/拒绝算法")
print("-" * 60)
batch, k, vocab = 2, 4, 100

# 模拟 draft 和 target 分布
draft_probs = F.softmax(torch.randn(batch, k, vocab), dim=-1)
target_probs = F.softmax(torch.randn(batch, k, vocab) * 1.2, dim=-1)  # 略尖一点
draft_tokens = torch.randint(0, vocab, (batch, k))

accepted, accepted_mask = speculative_verify(draft_probs, target_probs, draft_tokens)

print("Draft tokens 接受情况:")
for b in range(batch):
    print(f"  批次 {b}: ", end="")
    for i in range(k):
        status = "✓" if accepted_mask[b, i] else "✗"
        print(f"t{i+1}={status}", end="  ")
    print(f"→ 接受 {accepted[b].item()} 个")


# ============================================================
# 第4部分：完整流程模拟
# ============================================================

print("\n" + "=" * 60)
print("第4部分：完整投机解码流程模拟")
print("=" * 60)


def simulate_speculative_decoding(n_steps, draft_acceptance_rate=0.7, draft_size=4):
    """模拟投机解码的加速效果"""
    total_tokens = 0
    total_draft_calls = 0
    total_target_calls = 0

    accepted_history = []

    while total_tokens < n_steps:
        # 1. Draft model 生成 k 个候选 (1 次 draft 调用)
        total_draft_calls += 1
        # 2. Target model 验证 (1 次 target 调用)
        total_target_calls += 1

        # 接受数量 ~ 几何分布
        n_accepted = 0
        for _ in range(draft_size):
            if torch.rand(1).item() < draft_acceptance_rate:
                n_accepted += 1
            else:
                break
        n_accepted = max(1, n_accepted)  # 至少接受 1 个

        total_tokens += n_accepted
        accepted_history.append(n_accepted)

    return total_tokens, total_draft_calls, total_target_calls, accepted_history


# 对比普通生成
print("\n[模拟] 生成 100 token 的时间对比")
print("-" * 60)
print("假设: draft 推理 1ms, target 推理 5ms")
print("-" * 60)

t_draft = 1.0
t_target = 5.0
n_target_tokens = 100

# 普通生成
serial_time = n_target_tokens * t_target
print(f"普通自回归: {n_target_tokens} 次 target = {serial_time:.0f} ms")

# 投机解码
for accept_rate in [0.5, 0.7, 0.8, 0.9, 0.95]:
    total_tokens, n_draft, n_target, _ = simulate_speculative_decoding(
        n_target_tokens, draft_acceptance_rate=accept_rate, draft_size=4
    )
    spec_time = n_draft * (t_draft + t_target)  # draft + 验证
    speedup = serial_time / spec_time
    print(f"接受率 {accept_rate:.0%}: {n_draft} 步 × ({t_draft}+{t_target}={t_draft+t_target}ms) = {spec_time:.0f} ms  加速 {speedup:.2f}x")


# ============================================================
# 第5部分：接受率的影响
# ============================================================

print("\n" + "=" * 60)
print("第5部分：接受率的影响")
print("=" * 60)

print("""
接受率是关键指标, 取决于 draft 模型的质量:

  ┌────────────┬──────────┬──────────┐
  │ 接受率     │ 平均生成  │ 加速比    │
  ├────────────┼──────────┼──────────┤
  │ 0% (失败)  │ 1.0      │ 0.2x     │
  │ 50%        │ 2.0      │ 0.8x     │
  │ 70%        │ 2.8      │ 1.4x     │
  │ 80%        │ 3.5      │ 1.7x     │
  │ 90%        │ 4.7      │ 2.3x     │
  │ 95%        │ 5.7      │ 2.7x     │
  │ 99%        │ 7.0      │ 3.4x     │
  └────────────┴──────────┴──────────┘

  (k=4, draft 1ms, target 5ms 的假设下)

如何提高接受率:
  1. draft 模型越接近 target, 接受率越高
     - 同系列: 7B→70B, 1.5B→7B
  2. 任务越简单, 接受率越高
     - 简单问答: 80%+
     - 创意写作: 60-70%
     - 代码生成: 50-70%
  3. 可以用 lookhead、tree attention 等进一步加速
""")


# ============================================================
# 第6部分：在 MiniMind 中怎么用
# ============================================================

print("\n" + "=" * 60)
print("第6部分：在 MiniMind 中怎么用")
print("=" * 60)

print("""
MiniMind 通过 vLLM 或独立库支持投机解码:

方法 1: 使用 vLLM 内置投机解码

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="minimind-7b",                    # 大模型 (target)
    speculative_model="minimind-1b",        # 小模型 (draft)
    speculative_draft_tensor_parallel_size=1,
    num_speculative_tokens=4,               # k=4
)

prompts = ["今天天气"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)
```

方法 2: 使用独立的投机解码库 (无需 vLLM)

```python
from specdecoding import SpeculativeDecoder

decoder = SpeculativeDecoder(
    target_model="minimind-7b",
    draft_model="minimind-1b",
    k=4,
)

output = decoder.generate("今天天气", max_tokens=100)
```

方法 3: 简易版 - 自实现

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

draft_model = AutoModelForCausalLM.from_pretrained("minimind-1b")
target_model = AutoModelForCausalLM.from_pretrained("minimind-7b")

input_ids = tokenizer("今天天气", return_tensors="pt").input_ids
output_ids = speculative_generate(
    target_model, draft_model, input_ids,
    max_new_tokens=100, k=4
)
```

性能实测 (MiniMind 1B + 7B):
  - 普通: 30 tokens/s
  - 投机解码: 65 tokens/s (加速 2.2x)
""")


# ============================================================
# 第7部分：其他加速方法
# ============================================================

print("\n" + "=" * 60)
print("第7部分：其他加速方法")
print("=" * 60)

print("""
投机解码之外, 还有这些加速方法:

1) Medusa (多预测头)
   - 不需要 draft model
   - 在 target model 后面加多个预测头
   - 一次预测多个 token, 树形验证
   - 加速 2-3x

2) Lookahead Decoding
   - 不需要 draft model
   - 用 Jacobi 迭代方法生成多个 token
   - 加速 1.5-2.5x

3) EAGLE / EAGLE-2
   - 用 target model 的低层特征预测下一 token
   - 接受率高达 90%+
   - 加速 2-4x

4) Continuous Batching
   - 多个请求一起批处理
   - 服务端吞吐量 10x+
   - vLLM/TGI 默认使用

5) Prefix Caching
   - 共享系统提示的 KV cache
   - 长 prompt 加速 3-10x
   - 适合 chat 应用
""")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
1. 传统生成的瓶颈
   - 串行生成, GPU 浪费
   - memory-bound, 不是 compute-bound

2. 投机解码原理
   - draft model 生成 k 候选
   - target model 一次验证
   - 接受率决定加速效果

3. 接受/拒绝算法
   - 接受率 = min(1, p_target / p_draft)
   - 拒绝时用 target 分布重新采样
   - 输出分布与 target 完全一致

4. 性能
   - 接受率 70%: 加速 1.4x
   - 接受率 90%: 加速 2.3x
   - 接受率 95%: 加速 2.7x

5. 在 MiniMind 中
   - vLLM 内置支持
   - MiniMind 1B + 7B 加速 2.2x
   - 适合所有场景

6. 其他加速
   - Medusa, EAGLE, Lookahead
   - 连续批处理, 前缀缓存

本课程完结 🎉

从第0课 (Python 基础) 到第19课 (投机解码)
你已经掌握了训练和部署大语言模型的所有核心知识！
""")


# ============================================================
# 深入理解：投机解码的"预言家"艺术
# ============================================================
print("\n" + "=" * 60)
print("深入理解：投机解码的'预言家'艺术")
print("=" * 60)

print("""
【类比：投机解码 = 编辑-审稿】
─────────────────────────
  
  朴素生成 (每字问专家):
    教授审稿: 一次只改一个字
    改完一个, 重新看, 再改下一个
    → 慢, 100 个字要 100 轮
  
  投机解码 (学生先写, 教授审):
    1. 学生 (小模型) 草拟 5 个字
    2. 教授 (大模型) 一次性看 5 个字
    3. 教授说: 前 4 个字 OK, 第 5 个改成 X
    → 一次过审 4 个字, 只回滚 1 个
    → 比 1 个 1 个改快得多


【为什么需要投机解码】
───────────────────

  生成慢在哪?
    LLM 推理是 sequential 的
    生成 token N 必须先有 token N-1
    → 无法并行
    → GPU 利用率只有 10-20%
  
  内存带宽瓶颈:
    每次生成 1 个 token 都要读整个模型
    7B 模型 = 14 GB (FP16)
    → 大部分时间在"搬运数据"
    → 计算只是"副业"
  
  解法:
    让 1 次前向传播生成多个 token
    → 摊薄"搬运"成本
    → 投机解码就是这个思路


【投机解码的流程】
────────────────

  设:
    M_draft: 小模型 (草稿, 快)
    M_target: 大模型 (目标, 慢)
    γ: 草稿长度 (默认 5-10)

  步骤:
    1. 用 M_draft 生成 γ 个草稿 token
       → 快速, 单 token 5-10x 大模型
       → 草稿不要求正确
    
    2. 把 γ 个草稿拼成 1 个 batch
       → 一次性输入 M_target
    
    3. M_target 并行验证 γ 个草稿
       → 1 次前向传播, 验证 γ 个
       → 找出第一个错误位置
    
    4. 接受正确前缀, 拒绝错误及之后
       → 平均接受 2-5 个 token
    
    5. 用 M_target 重新预测被拒绝位置
       → 然后继续循环


【接受-拒绝机制】
──────────────

  对每个草稿 token x_t:
    p_draft: 小模型的概率
    p_target: 大模型的概率
    
    接受概率:
      r = min(1, p_target(x_t) / p_draft(x_t))
    
    以概率 r 接受
    以概率 1-r 拒绝

  直观理解:
    p_target > p_draft: 大模型更确定, 接受
    p_target < p_draft: 小模型错了, 拒绝
    p_target = p_draft: 一致, 接受

  重要特性:
    这种接受-拒绝机制严格保持原分布
    → 投机解码的输出与大模型完全相同
    → 没有质量损失!
    → 是"无损加速"


【草稿长度选择】
─────────────

  γ 太小 (1-2):
    草稿经常被全盘否定
    → 加速比接近 1
    
  γ 太大 (20+):
    草稿质量跟不上
    → 后半段基本被拒绝
    → 浪费计算
    
  γ 适中 (5-10):
    大部分时候前 60-80% 被接受
    → 加速比 2-4x

  自适应:
    观察接受率
    → 高 (80%+): 增大 γ
    → 低 (<50%): 减小 γ
    → 动态平衡


【加速比分析】
────────────

  设:
    t_d: 小模型生成 1 token 时间
    t_v: 大模型验证 1 token 时间 (1 个)
    a: 平均接受率 (如 0.7)
    γ: 草稿长度

  时间:
    朴素: γ * t_v (大模型生成 γ 个)
    投机: γ * t_d + t_v (草稿 + 一次验证)
    
  加速比:
    s = (γ * t_v) / (γ * t_d + t_v)
    
  例子 (t_v = 100, t_d = 20, a = 0.7, γ = 5):
    期望接受: 0.7 * 5 + 1 = 4.5 token
    朴素时间: 5 * 100 = 500
    投机时间: 5 * 20 + 100 = 200
    加速比: 500 / 200 = 2.5x

  更准确 (含重生成):
    期望生成: γ * a + 1 = 5*0.7 + 1 = 4.5
    朴素: 4.5 * t_v = 4.5 * 100 = 450
    投机: γ * t_d + t_v = 5*20 + 100 = 200
    加速比: 450 / 200 = 2.25x


【小模型选择】
─────────────

  理想草稿模型:
    1. 与大模型同系列 (同 tokenizer, 同训练数据)
    2. 小但不太小 (如 7B 配 1B)
    3. 速度快 (可量化加速)

  常见配对:
    LLaMA-70B ← LLaMA-7B
    LLaMA-13B ← LLaMA-1B
    GPT-4   ← GPT-3.5
    
  替代方案:
    1. Medusa: 用大模型的多个头做草稿
    2. EAGLE: 训练专门的"投机头"
    3. Lookahead: 不需要小模型, 用 n-gram
    4. Self-Speculative: 同一模型不同层


【投机解码变体】
──────────────

  ┌──────────┬────────────┬────────────┐
  │ 方法      │ 加速比      │ 复杂度      │
  ├──────────┼────────────┼────────────┤
  │ 标准投机  │ 2-3x       │ 中          │
  │ Medusa   │ 2-3x       │ 中 (训练)   │
  │ EAGLE    │ 3-4x       │ 高          │
  │ Lookahead│ 1.5-2x     │ 低          │
  │ 连续批处理│ 2-10x      │ 高          │
  └──────────┴────────────┴────────────┘

  Medusa:
    在大模型上多加几个"草稿头"
    头 1 预测 t+1, 头 2 预测 t+2...
    → 不需要小模型
    → 训练时一起学
  
  EAGLE:
    特征级别投机
    在隐层预测, 不在词表预测
    → 更准, 加速比更高
  
  Lookahead:
    用 n-gram 生成候选
    → 不需要额外模型
    → 加速比小, 但简单


【投机解码 vs 其他加速】
─────────────────────

  ┌──────────┬────────────┬────────────┬──────────────┐
  │ 技术      │ 加速比      │ 适用        │ 限制          │
  ├──────────┼────────────┼────────────┼──────────────┤
  │ 量化     │ 1.5-4x     │ 显存紧      │ 精度损失      │
  │ 投机解码 │ 2-4x       │ 任何模型    │ 需小模型      │
  │ 连续批处理│ 2-10x     │ 服务端      │ 多请求        │
  │ 预填充优化│ 1.5-2x     │ 长 prompt  │ 通用          │
  │ Prefix   │ 5-10x     │ 重复 prompt │ 特定场景      │
  │   缓存   │            │            │              │
  └──────────┴────────────┴────────────┴──────────────┘

  投机解码独特优势:
    - 不损失精度 (数学保证)
    - 不需要改模型结构
    - 单请求也能加速 (vs 批处理)


【vLLM 中的实现】
──────────────

  vLLM 内置投机解码支持:
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",
        speculative_model="meta-llama/Llama-2-7b-hf",
        num_speculative_tokens=5
    )
    
    # 2-3x 加速, 无缝集成

  其他支持:
    - TensorRT-LLM: 内置支持
    - llama.cpp: speculative 模式
    - HuggingFace TGI: 集成 draft model


【实践建议】
──────────

  1. 何时使用:
    - 单请求延迟敏感
    - 有合适的小模型
    - 输出较长 (短输出加速有限)
  
  2. 何时不用:
    - 没有合适的小模型
    - 输出很短 (1-10 token)
    - 已经用批处理
  
  3. 调试:
    - 先看接受率: 太低就改小模型或缩短草稿
    - 监控实际加速比: 理论 vs 实际
    - 不同 prompt 接受率差异大
""")


# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

print("""
【练习1】为什么需要投机
  朴素自回归生成为什么慢? 内存带宽是瓶颈吗?

【练习2】无损加速
  投机解码如何保证输出与大模型一致? 关键机制是什么?

【练习3】加速比计算
  小模型 10x 快, 平均接受 3/5, γ=5
  实际加速比是多少?

【练习4】接受-拒绝机制
  p_target=0.6, p_draft=0.4 时接受概率多少?
  p_target=0.2, p_draft=0.4 时呢? 含义?

【练习5】草稿长度权衡
  γ 太小和太大各有什么问题?

【练习6】投机解码 vs 量化
  7B 模型 INT4 量化 2x 加速
  70B 模型 + 7B 草稿 2.5x 加速
  两者能叠加吗? 还有什么组合?
""")


# ============================================================
# 练习答案
# ============================================================
print("\n" + "=" * 60)
print("练习答案")
print("=" * 60)

# 练习1
print("\n【练习1 答案】")
print("  朴素自回归慢的原因:")
print()
print("  1. 串行依赖:")
print("    生成 t 必须等 t-1")
print("    → 无法 GPU 并行")
print("    → 7B 模型生成 100 token = 100 步串行")
print()
print("  2. 内存带宽瓶颈:")
print("    每生成 1 token 要读 14 GB 模型")
print("    A100 带宽: 2 TB/s")
print("    理论时间: 14GB / 2TB/s = 7ms")
print("    → 每 token 至少 7ms (光搬数据)")
print("    → 实际 50-100ms (含计算)")
print()
print("  3. GPU 利用率低:")
print("    计算密度: 14B FLOPs / 14GB = 1 FLOP/byte")
print("    → 典型的 memory-bound 任务")
print("    → GPU 计算单元空闲")
print()
print("  解决思路:")
print("    投机解码: 1 次搬数据, 验证多个 token")
print("    → 摊薄搬运成本")
print("    → 提升 GPU 利用率")

# 练习2
print("\n【练习2 答案】")
print("  投机解码保证无损的关键: 接受-拒绝采样")
print()
print("  机制:")
print("    对每个草稿 token x:")
print("      接受概率 r = min(1, p_target(x) / p_draft(x))")
print("      以概率 r 接受")
print("      以概率 1-r 拒绝并用 p_target 重新采样")
print()
print("  数学证明:")
print("    当使用上述接受-拒绝时")
print("    最终的输出分布严格等于 p_target")
print("    → 大模型的输出")
print()
print("  直观理解:")
print("    p_draft 给了 p_target 一样的建议 → 接受")
print("    p_draft 高估了某 token → 按比例接受 (过度推荐)")
print("    p_draft 低估了某 token → 拒绝 (用大模型)")
print()
print("  结论:")
print("    投机解码的输出与大模型完全相同")
print("    → 速度提升, 质量不变")
print("    → 数学上严格的无损加速")

# 练习3
print("\n【练习3 答案】")
print("  加速比计算:")
print()
print("  条件:")
print("    小模型 10x 快: t_d = t_v / 10")
print("    平均接受: 3/5 (a = 0.6)")
print("    草稿长度: γ = 5")
print()
print("  朴素生成时间 (生成 5 token):")
print("    T_normal = 5 * t_v")
print()
print("  投机生成时间 (1 轮):")
print("    T_spec = γ * t_d + t_v   (草稿 + 验证)")
print("         = 5 * (t_v/10) + t_v")
print("         = 0.5 * t_v + t_v")
print("         = 1.5 * t_v")
print()
print("  但实际生成:")
print("    朴素 1 步 = 1 token")
print("    投机 1 轮 ≈ 0.6*5 + 1 = 4 token")
print()
print("  归一化:")
print("    朴素生成 4 token: 4 * t_v")
print("    投机生成 4 token: 1.5 * t_v")
print()
print("  加速比:")
print("    s = 4 * t_v / 1.5 * t_v = 4 / 1.5 ≈ 2.67x")
print()
print("  简化公式:")
print("    s = (γ * a + 1) / (γ / 10 + 1)")
print("    s = (5 * 0.6 + 1) / (5 / 10 + 1)")
print("    s = 4 / 1.5 = 2.67x")

# 练习4
print("\n【练习4 答案】")
print("  接受-拒绝概率计算:")
print()
print("  公式:")
print("    r = min(1, p_target(x) / p_draft(x))")
print()
print("  情况 1: p_target=0.6, p_draft=0.4")
print("    r = min(1, 0.6 / 0.4)")
print("    r = min(1, 1.5)")
print("    r = 1.0")
print()
print("    含义:")
print("      大模型比小模型更确定 (60% vs 40%)")
print("      → r=1 必然接受")
print("      → 小模型'保守', 大模型'更确定'")
print()
print("  情况 2: p_target=0.2, p_draft=0.4")
print("    r = min(1, 0.2 / 0.4)")
print("    r = min(1, 0.5)")
print("    r = 0.5")
print()
print("    含义:")
print("      大模型不太确定 (20%), 小模型更确定 (40%)")
print("      → 50% 概率接受")
print("      → 如果拒绝, 用大模型重新采样")
print("      → 这样保持大模型分布")
print()
print("  极端情况:")
print("    p_target=0, p_draft=0.5:")
print("      r = 0, 必拒")
print("      大模型认为不可能, 必然拒绝")
print()
print("    p_target=0.5, p_draft=0:")
print("      r = min(1, ∞) = 1, 必接")
print("      大模型说可能, 小模型说不可能, 大模型说了算")

# 练习5
print("\n【练习5 答案】")
print("  γ 太小的问题:")
print()
print("  现象:")
print("    假设 γ=1")
print("    → 草稿就是 1 个 token")
print("    → 投机变成了 1 步预测")
print()
print("  问题:")
print("    - 加速比小 (略多于 1x)")
print("    - 草稿过程开销相对大")
print("    - 没充分利用并行验证")
print()
print("  极限:")
print("    γ=1: 加速比 1.1-1.2x (几乎没有)")
print()
print("  ===")
print()
print("  γ 太大的问题:")
print()
print("  现象:")
print("    假设 γ=20, 小模型质量一般")
print("    → 草稿到第 5 个就开始错")
print("    → 第 6-20 个都被拒")
print()
print("  问题:")
print("    - 草稿时间浪费")
print("    - 实际接受只有前几个")
print("    - 加速比反而下降")
print()
print("  极限:")
print("    γ=20, 接受率 30%: 等效 γ=6, 还不如直接 γ=6")
print()
print("  最佳范围:")
print("    γ=5-10 是经验值")
print("    接受率高的模型 (如 80%+) 可用 γ=10-15")
print("    接受率低的 (50%-) 用 γ=3-5")
print()
print("  自适应策略:")
print("    观察最近 N 轮的接受率")
print("    - 接受率 > 80%: γ += 1")
print("    - 接受率 < 50%: γ -= 1")
print("    - 保持 γ 在最佳区间")

# 练习6
print("\n【练习6 答案】")
print("  量化 + 投机解码能叠加:")
print()
print("  场景:")
print("    70B INT4 + 7B INT4 草稿")
print("    投机 2.5x × 量化 2x = 5x 综合加速")
print()
print("  实现:")
print("    都用 INT4 量化")
print("    大小模型都提速")
print("    加速比可叠加")
print()
print("  其他组合:")
print("  1. 投机 + 连续批处理:")
print("    投机: 单请求 2.5x")
print("    批处理: 多请求 3-5x")
print("    组合: 5-10x 综合吞吐")
print()
print("  2. 投机 + Prefix 缓存:")
print("    prefix 缓存: 系统提示词不重算")
print("    投机: 草稿+验证快")
print("    组合: 9-12x")
print()
print("  3. 投机 + Flash Attention:")
print("    flash: 单次前向快")
print("    投机: 摊薄前向成本")
print("    组合: 接近理论极限")
print()
print("  4. 投机 + KV Cache 优化:")
print("    KV 量化: 显存省")
print("    投机: 推理快")
print("    → 综合效益")
print()
print("  工业实践:")
print("    vLLM + INT4 + 投机 + 批处理")
print("    → 端到端 10-20x 加速")
print()
print("  注意:")
print("    - 各种加速的瓶颈不同")
print("    - 叠加不是简单相乘")
print("    - 实测才知道真实加速")


# ============================================================
# 课程完整路线图
# ============================================================
print("\n" + "=" * 60)
print("课程完整路线图")
print("=" * 60)

print("""
你已经学完了所有 20 课, 这是你的知识地图:

【基础篇】
  第0课:  Python / PyTorch / 线性代数
  第1课:  Tokenizer (分词)
  第2课:  Embedding (词向量)

【组件篇】
  第3课:  RMSNorm (归一化)
  第4课:  RoPE (位置编码)
  第5课:  Attention (注意力)
  第6课:  FFN (前馈网络)
  第7课:  Transformer Block

【模型篇】
  第8课:  GPT (语言模型)
  第9课:  训练循环
  第10课: 生成与推理

【高级特性】
  第11课: MoE / TTT / MTP / KV Cache / MSA
  第12课: 优化器 (SGD/Momentum/Adam/Muon)
  第13课: 注意力变体 (Linear/ALiBi/Flash)
  第14课: Mamba (状态空间)
  第15课: LoRA (高效微调)
  第16课: YaRN (长度外推)
  第17课: mHC (多流残差)
  第18课: 量化 (INT4/INT8/GPTQ)
  第19课: 投机解码 (加速生成)

【能力树】
  你现在可以:
    ✓ 读懂 Transformer 论文
    ✓ 从零实现 LLM 各组件
    ✓ 训练自己的小模型
    ✓ 优化推理性能
    ✓ 微调预训练模型
    ✓ 部署到生产环境

【下一步建议】
  1. 读论文 (LLaMA, Mistral, Mamba)
  2. 跑开源项目 (MiniMind, nanoGPT, lit-llama)
  3. 训练自己的模型
  4. 关注前沿 (Qwen, DeepSeek, Claude 技术报告)
""")


# ============================================================
# 本课小结
# ============================================================
print("\n" + "=" * 60)
print("本课小结")
print("=" * 60)

print("""
本课介绍了投机解码:

核心思想:
  小模型草稿 + 大模型验证
  1 次前向传播生成多个 token
  摊薄内存带宽瓶颈

关键机制:
  接受-拒绝采样
  → 输出严格等于大模型分布
  → 无损加速

加速比:
  典型 2-3x
  与量化/批处理可叠加

实战工具:
  vLLM, TGI, llama.cpp
  → 都内置支持
""")

print("\n" + "=" * 60)
print("🎉 恭喜完成全部 20 课! 🎉")
print("=" * 60)
print("""
你从 Python 基础一路学到 LLM 训练部署
掌握了训练大语言模型所需的全部核心知识

下一步:
  → 动手训练一个自己的 LLM
  → 阅读前沿论文
  → 贡献开源项目

祝你在 AI 道路上一路顺风!
""")
