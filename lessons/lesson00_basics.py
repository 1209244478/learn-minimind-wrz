"""
第0课：0基础入门 — 准备工作
=============================

在开始学习大模型之前，你需要掌握三个最基础的技能：
  1. Python 基础语法
  2. PyTorch 张量操作
  3. 一点点线性代数

本课不涉及任何模型知识，只补齐基础。已经有基础的同学可以跳过。

运行: python lessons/lesson00_basics.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 第1部分：Python 基础
# ============================================================

print("=" * 60)
print("第1部分：Python 基础")
print("=" * 60)

print("""
Python 是一门简洁的编程语言。我们用几个例子快速过一遍你需要知道的内容。
""")

# 1. 变量和数据类型
print("[1.1] 变量和数据类型")
print("-" * 40)

name = "MiniMind"      # 字符串 str
age = 1                # 整数 int
pi = 3.14              # 浮点数 float
is_active = True       # 布尔值 bool
nothing = None         # 空值 NoneType

print(f"字符串: {name} (类型: {type(name).__name__})")
print(f"整数:   {age} (类型: {type(age).__name__})")
print(f"浮点数: {pi} (类型: {type(pi).__name__})")
print(f"布尔值: {is_active} (类型: {type(is_active).__name__})")
print(f"空值:   {nothing} (类型: {type(nothing).__name__})")


# 2. 列表和字典
print("\n[1.2] 列表 (List) 和字典 (Dict)")
print("-" * 40)

tokens = [1, 5, 10, 15, 20]
print(f"tokens = {tokens}")
print(f"第一个元素: tokens[0] = {tokens[0]}")
print(f"最后一个元素: tokens[-1] = {tokens[-1]}")
print(f"切片 tokens[1:3] = {tokens[1:3]}")
print(f"长度: len(tokens) = {len(tokens)}")

# 添加元素
tokens.append(25)
print(f"添加后: {tokens}")

# 字典：键值对
config = {
    "hidden_size": 512,
    "num_layers": 8,
    "vocab_size": 6400,
}
print(f"\nconfig = {config}")
print(f"隐藏层大小: config['hidden_size'] = {config['hidden_size']}")
print(f"所有键: {list(config.keys())}")


# 3. 循环和条件
print("\n[1.3] 循环 (for) 和条件 (if)")
print("-" * 40)

print("for 循环遍历列表:")
for token in tokens:
    print(f"  token = {token}")

print("\nfor 循环带索引:")
for i, token in enumerate(tokens[:3]):
    print(f"  第 {i} 个: {token}")

print("\nif 条件判断:")
score = 0.95
if score > 0.9:
    print(f"  分数 {score} > 0.9, 模型很确定！")
elif score > 0.5:
    print(f"  分数 {score} > 0.5, 模型有点把握")
else:
    print(f"  分数 {score} 太低, 模型不确定")


# 4. 函数
print("\n[1.4] 函数 (def)")
print("-" * 40)

def greet(name, language="中文"):
    """问候函数 - 文档字符串说明函数用途"""
    if language == "中文":
        return f"你好, {name}!"
    else:
        return f"Hello, {name}!"

print(greet("小白"))
print(greet("小白", language="English"))


# 5. 类 (Class)
print("\n[1.5] 类 (Class) - 面向对象")
print("-" * 40)

class SimpleTokenizer:
    """简单的分词器 - 演示类怎么写"""

    def __init__(self, vocab):
        """初始化：构建词表"""
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}

    def encode(self, text):
        """文本 → token IDs"""
        return [self.token_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        """token IDs → 文本"""
        return "".join([self.vocab[i] for i in ids])

vocab = ["<pad>", "你", "好", "世", "界", " "]
tokenizer = SimpleTokenizer(vocab)

ids = tokenizer.encode("你好 世界")
print(f"编码: 你好 世界 -> {ids}")
print(f"解码: {ids} -> '{tokenizer.decode(ids)}'")


# ============================================================
# 第2部分：PyTorch 基础
# ============================================================

print("\n" + "=" * 60)
print("第2部分：PyTorch 基础")
print("=" * 60)

print("""
PyTorch 是目前最流行的深度学习框架。它的核心数据结构是"张量"(Tensor)。
你可以把它理解为多维数组，和 NumPy 很像，但能在 GPU 上跑、能自动求导。
""")


# 1. 创建张量
print("[2.1] 创建张量 (Tensor)")
print("-" * 40)

# 从 Python 列表创建
t1 = torch.tensor([1, 2, 3, 4])
print(f"从列表创建: {t1}, shape={t1.shape}, dtype={t1.dtype}")

# 全 0 张量
t2 = torch.zeros(2, 3)
print(f"全0张量:\n{t2}\nshape={t2.shape}")

# 全 1 张量
t3 = torch.ones(2, 3)
print(f"全1张量:\n{t3}\nshape={t3.shape}")

# 随机张量
t4 = torch.randn(2, 3)
print(f"随机张量 (正态分布):\n{t4}\nshape={t4.shape}")

# 特殊张量
t5 = torch.arange(0, 6).reshape(2, 3)  # 0,1,2,3,4,5 排成 2x3
print(f"arange+reshape:\n{t5}")


# 2. 张量的属性
print("\n[2.2] 张量的重要属性")
print("-" * 40)

x = torch.randn(2, 3, 4)
print(f"x.shape = {x.shape}    # 形状: (2, 3, 4)")
print(f"x.ndim  = {x.ndim}     # 维度数: 3")
print(f"x.dtype = {x.dtype}   # 数据类型: float32")
print(f"x.device = {x.device}  # 在哪个设备上 (CPU/GPU)")

# 维度含义
print("""
维度 (Dimension) 的含义:
  1维 = 向量: [1, 2, 3]              (一个词的特征)
  2维 = 矩阵: [[1,2], [3,4]]         (一句话的多个词)
  3维 = 张量: [[[...]]]              (一个batch的多句话)
  4维 = [batch, seq, heads, dim]     (多头的注意力)
""")


# 3. 张量运算
print("[2.3] 张量运算")
print("-" * 40)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"a = \n{a}")
print(f"b = \n{b}")
print(f"\n逐元素相加: a + b = \n{a + b}")
print(f"逐元素相乘: a * b = \n{a * b}")
print(f"矩阵乘法:   a @ b = \n{a @ b}")
print(f"标量相加:   a + 10 = \n{a + 10}")


# 4. 索引和切片
print("\n[2.4] 索引和切片")
print("-" * 40)

x = torch.arange(12).reshape(3, 4)
print(f"x = \n{x}")
print(f"x[0, :]    = {x[0, :]}    # 第0行")
print(f"x[:, 1]    = {x[:, 1]}    # 第1列")
print(f"x[1:, 2:]  = \n{x[1:, 2:]}  # 右下角 2x2")
print(f"x[0, 0]    = {x[0, 0]}    # 单个元素")


# 5. 形状变换
print("\n[2.5] 形状变换 (reshape, transpose, permute)")
print("-" * 40)

x = torch.arange(24)
print(f"原始: x.shape = {x.shape}")

y = x.reshape(2, 3, 4)
print(f"reshape(2,3,4): y.shape = {y.shape}")

z = y.transpose(0, 1)  # 交换第0和第1维
print(f"transpose(0,1): z.shape = {z.shape}")

print("""
常用变换:
  reshape:    改变形状, 数据不变 (要求元素总数相同)
  transpose:  交换两个维度
  permute:    重新排列所有维度
  squeeze:    去掉大小为1的维度
  unsqueeze:  增加一个大小为1的维度
""")


# 6. 广播 (Broadcasting)
print("[2.6] 广播 (Broadcasting)")
print("-" * 40)

a = torch.tensor([[1], [2], [3]])  # shape: (3, 1)
b = torch.tensor([10, 20, 30])     # shape: (3,)
c = a + b  # 广播: b 被复制3次变成 (3, 3)
print(f"a (3,1) + b (3,) =\n{c}\nshape={c.shape}")

print("""
广播规则:
  1. 从右往左对齐维度
  2. 维度大小相同 或 其中一个是 1 → 可以广播
  3. 大小是 1 的维度会被"复制"扩展
""")


# 7. 自动求导
print("\n[2.7] 自动求导 (Autograd)")
print("-" * 40)

print("""
神经网络的核心是"自动求导":
  1. 创建参数 (requires_grad=True)
  2. 前向传播计算损失
  3. loss.backward() 自动计算梯度
  4. optimizer.step() 更新参数
""")

# 演示
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # y = x^2
y.backward()  # dy/dx = 2x = 4
print(f"x = 2, y = x^2 = {y.item()}")
print(f"dy/dx = 2x = {x.grad.item()}  (理论值是 4)")

# 神经网络中常用的写法
w = torch.randn(3, 4, requires_grad=True)  # 权重
x = torch.randn(2, 3)                       # 输入
y = x @ w                                   # 前向传播: y = x @ w
loss = y.sum()                              # 简单损失
loss.backward()                             # 反向传播
print(f"\n参数 w 的梯度 shape: {w.grad.shape}")
print(f"梯度 = 权重 w 的'修正方向'")


# 8. 神经网络基础
print("\n[2.8] 神经网络基础 (nn.Module)")
print("-" * 40)

print("""
PyTorch 用 nn.Module 来组织网络层:
  1. 继承 nn.Module
  2. 在 __init__ 中定义层
  3. 在 forward 中定义数据流
""")

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)  # 输入10维, 输出20维
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)   # 输入20维, 输出5维

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = TinyModel()
x = torch.randn(4, 10)  # 4个样本, 每个10维
y = model(x)
print(f"输入 shape: {x.shape}")
print(f"输出 shape: {y.shape}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# 9. GPU 加速
print("\n[2.9] GPU 加速 (可选)")
print("-" * 40)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = x @ y
    print(f"GPU 可用! 计算在 {device} 上完成")
else:
    print("GPU 不可用, 将使用 CPU (不影响学习)")

print("""
在 CPU 和 GPU 之间移动数据:
  .to(device)     - 把数据放到指定设备
  .cpu()          - 移回 CPU
  .cuda()         - 移到 GPU (简写)
""")


# ============================================================
# 第3部分：线性代数快速回顾
# ============================================================

print("\n" + "=" * 60)
print("第3部分：线性代数快速回顾")
print("=" * 60)

print("""
大模型里到处是矩阵运算。这里快速过一遍你需要知道的内容。
""")

# 1. 向量
print("[3.1] 向量 (Vector)")
print("-" * 40)

v = torch.tensor([1.0, 2.0, 3.0])
print(f"向量 v = {v}")
print(f"维度 = {v.shape}")
print(f"一个词的特征就是一个向量: [0.1, -0.3, 0.5, ...]")


# 2. 矩阵乘法
print("\n[3.2] 矩阵乘法 (Matrix Multiplication)")
print("-" * 40)

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print(f"A = \n{A}")
print(f"B = \n{B}")
print(f"A @ B = \n{A @ B}")

print("""
矩阵乘法的几何意义:
  - (m, k) @ (k, n) → (m, n)
  - 内维度必须匹配 (k == k)
  - 可以理解为"线性变换"和"组合特征"

  nn.Linear 本质上就是矩阵乘法:
    y = x @ W^T + b
  其中 W 是权重矩阵, b 是偏置
""")


# 3. 点积
print("\n[3.3] 点积 (Dot Product)")
print("-" * 40)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
dot = (a * b).sum()
print(f"a = {a}")
print(f"b = {b}")
print(f"点积 a·b = {dot}")
print(f"  = 1*4 + 2*5 + 3*6 = 4+10+18 = 32")

print("""
点积的几何意义:
  - 衡量两个向量的"相似度"
  - 同方向 → 大正数
  - 反方向 → 大负数
  - 垂直 → 0

  Attention 机制的核心就是点积!
  Q · K 越大, Q 和 K 的内容越相关
""")


# 4. Softmax
print("\n[3.4] Softmax: 把任意数变成概率")
print("-" * 40)

x = torch.tensor([1.0, 2.0, 3.0])
probs = torch.softmax(x, dim=-1)
print(f"输入:    {x}")
print(f"Softmax: {probs}")
print(f"  所有值都是正数")
print(f"  所有值加起来 = {probs.sum():.4f} ≈ 1.0")
print(f"  最大的输入 → 最大的概率")

print("""
Softmax 公式:
  softmax(x_i) = exp(x_i) / sum(exp(x_j))

为什么需要 Softmax?
  - 模型的输出是任意实数 (logits)
  - 我们需要把它变成概率 (总和为1)
  - 还要"放大"最大值, 让选择更明确
""")


# ============================================================
# 准备工具
# ============================================================

print("\n" + "=" * 60)
print("准备工具")
print("=" * 60)

print(f"""
✅ Python 版本: {torch.__version__[:6]} (PyTorch)
✅ 你的 PyTorch 版本: {torch.__version__}
✅ CUDA 可用: {torch.cuda.is_available()}
""")

print("""
必备工具:
  - Python 3.8+
  - PyTorch 2.0+
  - 文本编辑器 (VSCode / PyCharm / Trae)
  - 一点点耐心 😊

可选:
  - GPU (训练更快, 但 CPU 也能学)
  - Jupyter Notebook (交互式运行)

下一步: 学习第1课 - Tokenizer!
""")
