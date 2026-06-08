# ============================================================================
# 第23课：知识蒸馏 (Knowledge Distillation)
# ============================================================================
"""
大模型能力强但推理慢，小模型推理快但能力弱。
知识蒸馏的目标：让小模型（学生）学会大模型（老师）的知识。

本课内容：
1. 为什么需要知识蒸馏？
2. 蒸馏的核心思想：软标签 vs 硬标签
3. KL 散度与蒸馏损失
4. 温度参数 (Temperature) 的作用
5. 蒸馏训练循环实现

关键概念：
- 教师模型 (Teacher)：大模型，提供软标签
- 学生模型 (Student)：小模型，学习软标签
- 软标签 (Soft Labels)：教师输出的概率分布
- 硬标签 (Hard Labels)：one-hot 编码的真实标签
- 温度 (Temperature)：控制软标签"软度"的参数
- KL 散度 (KL Divergence)：衡量两个分布差异的指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. 为什么需要知识蒸馏？
# ============================================================================
"""
问题场景：
  - GPT-4 能力强，但推理成本高（数千亿参数）
  - 部署到手机/边缘设备需要小模型（数亿参数）
  - 直接训练小模型效果差

知识蒸馏的解决方案：
  用大模型（教师）的输出作为"软标签"来训练小模型（学生）

直觉理解：
  传统训练：学生只看标准答案（硬标签）
    "这张图片是猫" → [0, 0, 1, 0, 0]  (one-hot)

  知识蒸馏：学生看老师的评分（软标签）
    "这张图片90%是猫，5%是狗，3%是老虎..." → [0.05, 0.03, 0.90, 0.01, 0.01]

  软标签包含更多信息！它告诉学生"猫和老虎很像"这种暗知识 (Dark Knowledge)。

【为什么软标签比硬标签好？一个具体的例子】

  假设我们在做图片分类，有5个类别：猫、狗、老虎、汽车、桌子

  硬标签（传统训练）：
    正确答案是"猫" → [0, 0, 1, 0, 0]
    学生只知道："答案是猫，其他都不对"
    学到的：猫 ≠ 狗，猫 ≠ 汽车（但不知道猫和谁更"像"）

  软标签（知识蒸馏，T=2）：
    教师输出 → [0.60, 0.10, 0.25, 0.03, 0.02]
    学生学到：
      - 猫的概率最高（60%）→ 答案是猫 ✓
      - 老虎的概率排第二（25%）→ 猫和老虎很像！（都是猫科）
      - 狗的概率排第三（10%）→ 猫和狗有点像！（都是宠物）
      - 汽车和桌子概率很低 → 猫和它们完全不像

  这就是"暗知识"：硬标签只告诉你"谁是正确的"，
  软标签还告诉你"谁和谁相似"——这些相似性信息非常有价值！

  类比：
    硬标签 = 考试只告诉你"选C"
    软标签 = 老师说"C最对，但B也有点道理，A完全不对"
    → 软标签让你理解了"为什么"，而不只是"选什么"
"""

print("=" * 70)
print("第23课：知识蒸馏 (Knowledge Distillation)")
print("=" * 70)

# ============================================================================
# 2. 软标签 vs 硬标签
# ============================================================================
"""
硬标签 (Hard Label)：
  真实类别 = 2 → [0, 0, 1, 0, 0, 0]
  只有正确类别为1，其余为0，没有额外信息

软标签 (Soft Label)：
  教师输出 → [0.05, 0.03, 0.85, 0.04, 0.02, 0.01]
  正确类别概率最高，但其他类别也有非零概率
  这些非零概率包含了类别间的相似性信息（暗知识）

温度 (Temperature) 的作用：
  softmax(z/T) — T 越大，分布越"软"（越均匀）

  T=1:  [0.05, 0.03, 0.85, 0.04, 0.02, 0.01]  ← 正常 softmax
  T=2:  [0.12, 0.09, 0.45, 0.14, 0.11, 0.09]  ← 更软，暗知识更明显
  T=10: [0.16, 0.16, 0.19, 0.17, 0.16, 0.16]  ← 非常软，接近均匀
"""

print("\n--- 软标签 vs 硬标签 ---")
logits = torch.tensor([2.0, 1.0, 5.0, 1.5, 0.5, 0.3])

# 硬标签
hard_label = F.one_hot(torch.tensor([2]), num_classes=6).float()[0]
print(f"硬标签: {hard_label.tolist()}")

# 不同温度的软标签
for T in [1, 2, 5, 10]:
    soft_label = F.softmax(logits / T, dim=0)
    print(f"软标签 (T={T:2d}): {[f'{p:.3f}' for p in soft_label.tolist()]}")

# ============================================================================
# 3. KL 散度与蒸馏损失
# ============================================================================


def distillation_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.5):
    """知识蒸馏损失

    L = α * L_hard + (1-α) * L_soft

    L_hard = CrossEntropy(student_logits, labels)         — 与真实标签的交叉熵
    L_soft = KL(student_soft || teacher_soft) * T²        — 与教师分布的 KL 散度

    Args:
        student_logits: 学生模型的 logits
        teacher_logits: 教师模型的 logits
        temperature: 蒸馏温度
        alpha: 硬标签损失的权重 (0~1)

    Returns:
        loss: 蒸馏损失
    """
    # 软标签损失：KL 散度
    # 注意：F.kl_div 的输入是 log_probs 和 target_probs
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL 散度：KL(teacher || student) = Σ teacher * log(teacher/student)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    # 乘以 T² 补偿温度缩放
    soft_loss = (temperature ** 2) * kl_loss

    return soft_loss


def full_distillation_loss(student_logits, teacher_logits, labels,
                           temperature=1.0, alpha=0.5):
    """完整的蒸馏损失（包含硬标签损失）

    L = α * L_hard + (1-α) * L_soft

    在语言模型蒸馏中，通常只用 soft_loss（alpha=0），
    因为语言模型的"硬标签"就是下一个 token 的交叉熵，
    已经包含在预训练中了。
    """
    # 硬标签损失
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # 软标签损失
    soft_loss = distillation_loss(student_logits, teacher_logits, temperature)

    return alpha * hard_loss + (1 - alpha) * soft_loss


# 演示蒸馏损失计算
print("\n--- 蒸馏损失计算示例 ---")
torch.manual_seed(42)
B, T, V = 2, 8, 100

student_logits = torch.randn(B, T, V)
teacher_logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))

for T_val in [1, 2, 4, 8]:
    loss = distillation_loss(student_logits, teacher_logits, temperature=T_val)
    print(f"  T={T_val}: soft_loss = {loss.item():.4f}")

# ============================================================================
# 4. 温度参数的深入理解
# ============================================================================
"""
温度 T 的作用：

1. T=1：正常 softmax，分布尖锐
   - 教师输出接近 one-hot
   - 暗知识不明显
   - 蒸馏效果有限

2. T>1：软化分布，暗知识更明显
   - 教师输出更平滑
   - 类别间相似性信息更丰富
   - 蒸馏效果更好

3. T 很大：分布接近均匀
   - 信息量太少
   - 蒸馏效果下降

为什么乘以 T²？
  softmax(z/T) 的梯度比 softmax(z) 小 T 倍
  乘以 T² 保证梯度量级与温度无关
  数学推导：∂L/∂z ∝ 1/T，所以需要 * T² 来补偿
"""

print("\n--- 温度参数影响 ---")
logits = torch.tensor([3.0, 1.0, 0.5, -1.0, -2.0])

# 计算不同温度下的信息熵（衡量分布的"信息量"）
for T in [0.5, 1, 2, 5, 10]:
    probs = F.softmax(logits / T, dim=0)
    entropy = -(probs * probs.log()).sum()
    print(f"  T={T:4.1f}: entropy={entropy.item():.3f}, probs={[f'{p:.3f}' for p in probs.tolist()]}")

print("  信息熵越大 → 分布越均匀 → 暗知识越明显")

# ============================================================================
# 5. 蒸馏训练循环
# ============================================================================


class TeacherModel(nn.Module):
    """教师模型（大模型）"""

    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


class StudentModel(nn.Module):
    """学生模型（小模型）"""

    def __init__(self, vocab_size, dim=32, n_layers=2, n_heads=2, max_len=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def train_distillation(teacher, student, dataset, epochs=5, lr=3e-4,
                       temperature=2.0, alpha=0.0):
    """知识蒸馏训练循环

    Args:
        teacher: 教师模型（冻结）
        student: 学生模型（训练）
        temperature: 蒸馏温度
        alpha: 硬标签损失权重（0=纯蒸馏，1=纯硬标签）
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    teacher.eval()  # 教师模型始终在 eval 模式

    student.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for input_ids, labels in loader:
            # 教师模型前向传播（不计算梯度）
            with torch.no_grad():
                teacher_logits = teacher(input_ids)

            # 学生模型前向传播
            student_logits = student(input_ids)

            # 计算蒸馏损失
            if alpha > 0:
                loss = full_distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=temperature, alpha=alpha
                )
            else:
                loss = distillation_loss(
                    student_logits, teacher_logits, temperature=temperature
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs}, Distill Loss: {avg_loss:.4f}")

    return student


# 创建蒸馏数据集
class SimpleDistillDataset:
    """蒸馏数据集

    使用字符级 Tokenizer 编码文本数据。
    """

    def __init__(self, texts=None, max_length=32):
        self.max_length = max_length
        # 默认语料
        if texts is None:
            texts = [
                "深度学习是机器学习的一个子集，使用多层神经网络自动学习特征表示。",
                "Transformer架构基于自注意力机制，能并行处理序列数据。",
                "知识蒸馏让小模型学会大模型的知识，实现模型压缩。",
                "Python是一种高级编程语言，以简洁易读著称。",
                "自然语言处理是人工智能的重要研究方向。",
            ]
        # 构建字符级词表
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        chars = sorted(all_chars)
        self.char2id = {c: i + 1 for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # +1 for pad
        self.pad_token_id = 0
        # 编码所有文本
        self.samples = []
        for text in texts:
            ids = [self.char2id.get(c, self.pad_token_id) for c in text]
            ids = ids[:self.max_length] + [self.pad_token_id] * max(0, self.max_length - len(ids))
            self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.samples[idx], dtype=torch.long)
        labels = input_ids.clone()
        labels[:10] = -100  # 前10个 token 不计算损失
        return input_ids, labels


# 运行蒸馏训练
print("\n--- 知识蒸馏训练演示 ---")
distill_dataset = SimpleDistillDataset(max_length=32)
vocab_size = distill_dataset.vocab_size

teacher = TeacherModel(vocab_size=vocab_size, dim=128, n_layers=4, n_heads=4, max_len=32)
student = StudentModel(vocab_size=vocab_size, dim=32, n_layers=2, n_heads=2, max_len=32)
teacher.requires_grad_(False)

teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e3
student_params = sum(p.numel() for p in student.parameters()) / 1e3
print(f"教师模型参数量: {teacher_params:.1f}K")
print(f"学生模型参数量: {student_params:.1f}K")
print(f"压缩比: {teacher_params / student_params:.1f}x")

print(f"词表大小: {vocab_size}")

print("\n[蒸馏训练开始] T=2.0, α=0.0 (纯蒸馏)")
student = train_distillation(teacher, student, distill_dataset,
                             epochs=5, lr=3e-4, temperature=2.0, alpha=0.0)

# ============================================================================
# 6. 蒸馏策略对比
# ============================================================================
print("\n--- 蒸馏策略对比 ---")
print("""
┌──────────────────┬──────────────────────────────────────────┐
│ 策略             │ 说明                                     │
├──────────────────┼──────────────────────────────────────────┤
│ 纯蒸馏 (α=0)     │ 只用教师软标签，适合有充足教师数据        │
│ 纯硬标签 (α=1)   │ 只用真实标签，等同普通训练                │
│ 混合 (α=0.5)     │ 软硬结合，通常效果最好                    │
│ 特征蒸馏         │ 对齐中间层特征，不只是输出层              │
│ 渐进蒸馏         │ 逐步减小温度，先学粗后学细                │
└──────────────────┴──────────────────────────────────────────┘
""")

# ============================================================================
# 7. 蒸馏的变体
# ============================================================================
"""
1. 输出层蒸馏（本课实现）
   - 只对齐最终输出层的 logits
   - 最简单，效果也不错

2. 中间层蒸馏
   - 对齐教师和学生的中间层表示
   - 需要设计映射层（因为维度不同）
   - 效果更好，但实现更复杂

3. 注意力蒸馏
   - 对齐教师和学生的注意力权重
   - 让学生学会教师的注意力模式

4. 渐进蒸馏
   - 先用大温度（T=8）学粗粒度知识
   - 再用小温度（T=2）学细粒度知识
   - 类似课程学习 (Curriculum Learning)

5. 在线蒸馏
   - 教师和学生同时训练
   - 不需要预训练好的教师模型
"""

# ============================================================================
# 练习
# ============================================================================
print("\n" + "=" * 70)
print("练习")
print("=" * 70)

# 练习1：手动计算 KL 散度
print("\n练习1：手动计算 KL 散度")
p = torch.tensor([0.8, 0.1, 0.1])  # 教师分布
q = torch.tensor([0.4, 0.3, 0.3])  # 学生分布
kl = F.kl_div(q.log(), p, reduction='sum')
print(f"教师分布 P = {p.tolist()}")
print(f"学生分布 Q = {q.tolist()}")
print(f"KL(P||Q) = {kl.item():.4f}")
print("注意：KL 散度不对称！KL(P||Q) ≠ KL(Q||P)")

# 练习2：温度对软标签的影响
print("\n练习2：温度对软标签的影响")
logits = torch.tensor([5.0, 1.0, 0.0, -1.0])
for T in [1, 2, 5]:
    soft = F.softmax(logits / T, dim=0)
    print(f"  T={T}: {soft.tolist()}")

# 练习3：为什么乘以 T²
print("\n练习3：为什么蒸馏损失要乘以 T²？")
print("答案：因为 softmax(z/T) 的梯度比 softmax(z) 小 T 倍，")
print("      乘以 T² 可以保证梯度量级与温度无关，")
print("      避免温度增大时梯度消失。")

# 练习4：模型压缩比
print("\n练习4：计算模型压缩比")
print(f"教师: dim=128, layers=4, heads=4 → {teacher_params:.1f}K 参数")
print(f"学生: dim=32, layers=2, heads=2 → {student_params:.1f}K 参数")
print(f"压缩比: {teacher_params / student_params:.1f}x")
print("如果想要 10x 压缩，学生模型应该怎么设计？")
print("提示：参数量 ∝ dim² * layers，所以 dim 减半约 4x，layers 减半约 2x")

print("\n" + "=" * 70)
print("第23课总结：")
print("  1. 知识蒸馏：让小模型（学生）学习大模型（教师）的知识")
print("  2. 软标签包含暗知识（类别间相似性），比硬标签信息更丰富")
print("  3. 温度 T 控制软标签的'软度'，T 越大分布越均匀")
print("  4. 蒸馏损失 = KL(教师分布 || 学生分布) * T²")
print("  5. 常用策略：纯蒸馏、混合蒸馏、特征蒸馏、渐进蒸馏")
print("=" * 70)
