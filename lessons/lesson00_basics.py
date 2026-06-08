"""
第0课：0基础入门 — 准备工作
=============================

在开始学习大模型之前，你需要掌握三个最基础的技能：
  1. Python 基础语法
  2. PyTorch 张量操作
  3. 一点点线性代数

本课不涉及任何模型知识，只补齐基础。已经有基础的同学可以跳过。

【环境安装】
  1. 安装 Python 3.8+：https://www.python.org/downloads/
     安装时勾选 "Add Python to PATH"（重要！）
  2. 打开终端（Windows: 按 Win+R 输入 cmd；Mac: 打开"终端"app）
  3. 安装 PyTorch：在终端输入 pip install torch
  4. 安装其他依赖：pip install numpy
  5. 验证安装：pip install torch numpy && python -c "import torch; print(torch.__version__)"
  6. 运行本课：python lessons/lesson00_basics.py

  如果你用的是 GPU，安装 GPU 版 PyTorch：
    pip install torch --index-url https://download.pytorch.org/whl/cu118
  （cu118 对应 CUDA 11.8，根据你的 CUDA 版本选择）

  常见问题：
    Q: "pip 不是内部命令" → Python 没加到 PATH，重新安装勾选
    Q: "No module named torch" → 没装 PyTorch，运行 pip install torch
    Q: 下载太慢 → 用国内镜像：pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

运行: python lessons/lesson00_basics.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 第0部分：5分钟理解深度学习
# ============================================================

print("=" * 60)
print("第0部分：5分钟理解深度学习")
print("=" * 60)

print("""
【什么是深度学习？用一句话说】

  深度学习 = 让计算机通过"看大量数据"自动学会一项技能

  类比：教小孩认猫
    传统编程：你写规则"如果有尖耳朵+胡须+尾巴 → 是猫"
             问题：规则写不完，总有漏的
    深度学习：给小孩看1万张猫的照片，他自己学会认猫
             优势：不需要写规则，自己总结规律

【大模型是什么？】

  大模型 = 读了几千亿字的文章后，学会了"说话"的深度学习模型

  训练过程：
    1. 给模型看大量文本："今天天气很___"
    2. 模型猜下一个字："好"（猜对了奖励，猜错了调整）
    3. 重复几万亿次 → 模型学会了语言的规律
    4. 你问它问题，它就能像人一样回答

【你需要学什么？】

  要理解大模型，你需要知道：
    1. Python — 和计算机沟通的语言（本课第1部分）
    2. 张量 (Tensor) — 计算机存储数据的方式（本课第2部分）
    3. 矩阵运算 — 大模型计算的核心（本课第3部分）
    4. 自动求导 — 让模型"自动学习"的魔法（本课第2部分）

  学完这4个，你就能看懂后面所有的课程了！
""")

# ============================================================
# 第0.5部分：终端和 pip 是什么？
# ============================================================

print("=" * 60)
print("第0.5部分：终端和 pip 是什么？")
print("=" * 60)

print("""
如果你从没用过终端，这里快速解释：

【终端 (Terminal / CMD / PowerShell)】
  就是一个黑色窗口，你输入文字命令，计算机执行。
  打开方式：
    Windows: 按 Win+R，输入 cmd，回车
    Mac: 打开"终端"应用
    VSCode/Trae: 菜单栏 → 终端 → 新建终端

【pip 是什么？】
  pip = Python 的"应用商店"
  就像手机用 App Store 安装 App，
  Python 用 pip 安装"包"（别人写好的代码库）。

  常用命令：
    pip install xxx    → 安装 xxx 包
    pip uninstall xxx  → 卸载 xxx 包
    pip list           → 查看已安装的包

【import 是什么？】
  import = 在你的代码里"引入"别人写好的包

  import torch       → 引入 PyTorch（深度学习框架）
  import numpy as np → 引入 NumPy（数学计算库），简写为 np

  类比：
    pip install = 去书店买一本书
    import      = 把书从书架拿下来翻开
    用里面的函数 = 照着书上的方法做
""")


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
[OK] Python version: {torch.__version__[:6]} (PyTorch)
[OK] Your PyTorch version: {torch.__version__}
[OK] CUDA available: {torch.cuda.is_available()}
""")

print("""
必备工具:
  - Python 3.8+
  - PyTorch 2.0+
  - 文本编辑器 (VSCode / PyCharm / Trae)
  - 一点点耐心

可选:
  - GPU (训练更快, 但 CPU 也能学)
  - Jupyter Notebook (交互式运行)
""")


# ============================================================
# 第4部分：把所有知识串起来 — 训练你的第一个模型
# ============================================================

print("=" * 60)
print("第4部分：把所有知识串起来 — 训练你的第一个模型")
print("=" * 60)

print("""
前面学了 Python、张量、矩阵运算、自动求导、nn.Module...
这些知识怎么串起来？答案就是：训练一个模型！

下面我们用 20 行代码，从零训练一个"预测数字"的小模型。
这是大模型训练的微缩版——原理完全一样，只是规模小了几万倍。
""")

# --- 第1步：准备数据 ---
print("[4.1] 第1步：准备数据")
print("-" * 40)

# 训练数据：输入 x，预测 y = x * 2 + 1
# 类比：大模型的训练数据是"大量文本"，这里是"大量数字对"
torch.manual_seed(42)
x_train = torch.randn(100, 1)           # 100个随机输入
y_train = x_train * 2 + 1 + torch.randn(100, 1) * 0.1  # y = 2x + 1 + 噪声

print(f"训练数据: {len(x_train)} 个样本")
print(f"前3个: x={x_train[:3].squeeze().tolist()}")
print(f"       y={y_train[:3].squeeze().tolist()}")
print(f"目标: 让模型学会 y ≈ 2x + 1 这个规律")


# --- 第2步：定义模型 ---
print("\n[4.2] 第2步：定义模型")
print("-" * 40)

class SimpleModel(nn.Module):
    """最简单的模型：1个输入 → 1个输出（就是 y = w*x + b）"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 1维输入，1维输出

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
print(f"模型: y = w*x + b")
print(f"初始参数: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")
print(f"（初始是随机值，还没学呢）")


# --- 第3步：定义损失函数和优化器 ---
print("\n[4.3] 第3步：定义损失函数和优化器")
print("-" * 40)

criterion = nn.MSELoss()  # 均方误差：衡量预测值和真实值的差距
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

print(f"损失函数: MSELoss (预测值和真实值的差的平方)")
print(f"优化器: SGD (学习率=0.01)")
print(f"类比:")
print(f"  损失函数 = 考试分数（越低越好）")
print(f"  优化器 = 学习方法（每次调整一点点）")


# --- 第4步：训练循环 ---
print("\n[4.4] 第4步：训练循环（核心！）")
print("-" * 40)

print("""
训练循环就是4步，反复执行：
  1. 前向传播：用当前参数算预测值
  2. 算损失：预测值和真实值差多少
  3. 反向传播：算梯度（参数该往哪调）
  4. 更新参数：按梯度方向调整参数
""")

for epoch in range(200):
    # 1. 前向传播
    y_pred = model(x_train)

    # 2. 算损失
    loss = criterion(y_pred, y_train)

    # 3. 反向传播（自动求导！）
    optimizer.zero_grad()  # 清空旧梯度
    loss.backward()        # 计算新梯度

    # 4. 更新参数
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, w={w:.4f}, b={b:.4f}")


# --- 第5步：看结果 ---
print("\n[4.5] 第5步：看结果")
print("-" * 40)

w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"训练前: w=随机, b=随机")
print(f"训练后: w={w:.4f}, b={b:.4f}")
print(f"真实值: w=2.0000, b=1.0000")
print(f"误差:   w差{abs(w-2):.4f}, b差{abs(b-1):.4f}")
print(f"→ 模型学会了 y ≈ {w:.2f}x + {b:.2f}，非常接近 y = 2x + 1！")


# --- 测试 ---
print("\n[4.6] 测试：给模型一个新输入")
print("-" * 40)

test_x = torch.tensor([[3.0]])
test_y = model(test_x).item()
true_y = 3.0 * 2 + 1
print(f"输入 x = 3.0")
print(f"模型预测: y = {test_y:.4f}")
print(f"真实值:   y = {true_y:.4f}")
print(f"误差: {abs(test_y - true_y):.4f}")


print("""
═══════════════════════════════════════════
【总结：这个 Demo 和大模型的关系】
═══════════════════════════════════════════

  这个小模型                  大模型 (GPT)
  ──────────                  ──────────
  输入: 1个数字               输入: 一段文字（几百个token）
  输出: 1个数字               输出: 下一个字的概率
  参数: 2个 (w, b)            参数: 几十亿个
  训练数据: 100个数字对        训练数据: 几千亿字
  训练循环: 200步              训练循环: 几百万步
  损失函数: MSELoss            损失函数: CrossEntropyLoss

  但核心流程完全一样！
    1. 准备数据
    2. 定义模型
    3. 定义损失函数和优化器
    4. 循环：前向→算损失→反向→更新
    5. 测试效果

  后面所有课程，都是在这个框架上"加东西"：
    - 更复杂的模型结构（Attention, Transformer, ...）
    - 更大的数据（文本语料）
    - 更多的训练技巧（学习率调度, 梯度裁剪, ...）

  但万变不离其宗，核心就是这5步！

下一步: 学习第1课 - Tokenizer（把文字变成数字）!
""")
