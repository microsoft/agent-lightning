# APO算法原理：什么是"训练提示词"？

## 核心概念：这不是训练模型，而是优化提示词！

### 🔑 关键区别

| 传统模型训练 | APO提示词优化 |
|------------|-------------|
| 修改模型的**权重参数** | 修改**提示词文本** |
| 需要大量GPU资源 | 只需要调用API |
| 模型文件在本地 | 模型在远程服务器 |
| 训练时间长（小时/天） | 优化时间短（分钟） |

## 📝 什么是"训练提示词"？

**"训练提示词"** = **"优化提示词模板"** = **"自动化的提示词工程"**

### 举个例子：

假设你有一个**基线提示词**（baseline prompt）：
```
"Find a room on {date} at {time} for {duration_min} minutes, {attendees} attendees. 
Needs: {needs}. Accessible required: {accessible_required}"
```

APO算法会：
1. **评估**这个提示词的效果（在验证集上测试，得到分数）
2. **生成变体**（通过LLM生成改进版本）：
   - 变体1: "You are a scheduling assistant. Find a room on {date}..."
   - 变体2: "Select the best meeting room on {date} at {time}..."
3. **测试每个变体**，看哪个效果更好
4. **保留最佳变体**，继续优化
5. **重复**这个过程，直到找到最优提示词

## 🔄 APO训练流程详解

### 步骤1：初始化基线提示词
```python
initial_resources={
    "prompt_template": prompt_template_baseline()  # 基线提示词
}
```

### 步骤2：评估基线效果
- 在验证集上测试当前提示词
- 每个任务调用API，得到结果
- 计算平均分数（reward）

### 步骤3：生成提示词变体（Beam Search）
```
Round 1:
  - 从基线提示词开始
  - 生成2个变体（branch_factor=2）
  - 测试每个变体，保留最好的2个（beam_width=2）

Round 2:
  - 从上一轮最好的2个提示词开始
  - 每个再生成2个变体（共4个）
  - 测试所有变体，保留最好的2个
  - ...
```

### 步骤4：使用梯度信息优化
- APO会分析哪些提示词效果好
- 使用"梯度"信息指导生成更好的变体
- 类似于梯度下降，但是针对文本提示词

## 🌐 为什么模型不在本地也能"训练"？

### 关键理解：

1. **模型本身不变**
   - DeepSeek/GPT模型在远程服务器上
   - 我们**不修改**模型的任何参数
   - 只是**调用API**，传入不同的提示词

2. **优化的是提示词文本**
   - 提示词是一个**字符串模板**
   - 我们不断尝试不同的文本
   - 找到能让模型表现最好的提示词

3. **训练过程**
   ```
   提示词v0 → 调用API → 得到结果 → 评分
   提示词v1 → 调用API → 得到结果 → 评分
   提示词v2 → 调用API → 得到结果 → 评分
   ...
   选择分数最高的提示词！
   ```

## 📊 从日志看训练过程

从 `apo.log` 可以看到：

```
[Round 00 | Prompt v0] Evaluating seed prompt...
  → 评估基线提示词v0

[Round 01] Applying 2 edits to each of the 2 parents...
  → 生成2个变体

[Round 01 | Beam 01 | Branch 01 | Prompt v0] Evaluating prompt...
  → 测试第一个变体

[Round 01 | Beam 01 | Branch 02 | Prompt v1] Evaluating prompt...
  → 测试第二个变体
```

## 🎯 最终目标

找到一个**最优的提示词模板**，使得：
- 模型能更准确地理解任务
- 在验证集上获得更高的分数
- 不需要修改模型本身

## 💡 类比理解

想象你在教一个**固定的学生**（模型）：
- ❌ 你不能改变学生的能力（模型权重）
- ✅ 但你可以**改进你的教学方法**（提示词）
- 通过尝试不同的教学方式，找到最有效的那一种

这就是APO算法在做的事情！

