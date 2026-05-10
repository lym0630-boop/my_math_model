# prefix_guided_rollout.py 详解

## 文件位置
`pipline/lppo/prefix_guided_rollout.py`

## 核心职责
为 "hard-zero" 题目（模型完全不会，p < 0.05）准备带参考解前缀的 prompt，让模型从中间步骤续写，从而获得正向 reward 信号。

---

## 动机：为什么需要 PG？

标准 GRPO 中，如果一道题 8 次 rollout 全错：
```
rewards = [0, 0, 0, 0, 0, 0, 0, 0]
advantage = (0 - mean(0)) / std(0) = NaN or 0
```

这道题对模型的梯度贡献为 **零**。LP weight 再怎么调也没用——因为 advantage 本身就是 0。

**PG 的思路**：给模型一个"脚手架"（前缀），让它从正确方向开始推理：
```
原始 prompt: "求 ∫sin(x)dx = ?"
PG prompt:   "求 ∫sin(x)dx = ?" + prefix: "让我使用分部积分法。设 u=sin(x), dv=dx..."
```

如果模型从前缀续写后得到正确答案，就获得了 reward=1 的正向信号。

---

## 与 warmstart 的关系

| 特性 | warmstart | PG rollout |
|------|-----------|------------|
| 阶段 | 训练前（SFT） | 训练中（GRPO） |
| 目的 | 做 SFT 微调 | 在 GRPO batch 中提供学习信号 |
| 前缀变体 | 3 种（20%/40%/60%） | 1 种（40%） |
| 过滤 | 多层过滤（正确性+相似度） | 无过滤（靠 reward 函数判断） |

**复用的函数**：
```python
from prefix_guided_warmstart import (
    find_semantic_breakpoints,  # 找语义切割点
    cut_prefix,                 # 按比例截取
    check_answer_leakage,       # 答案泄露检查
    truncate_before_leakage,    # 截短泄露
)
```

---

## 核心函数 `prepare_pg_prompt`

```python
def prepare_pg_prompt(question, reference_answer, ground_truth, prefix_ratio=0.40):
```

### 处理流程

```
1. 找语义断点
   reference_answer → find_semantic_breakpoints()
   断点类型：$$ 结尾、空行、列表项、加粗标题

2. 按比例截取前缀（40%）
   cut_prefix(text, 0.40, breakpoints)
   在 40% 位置附近找最近的语义断点切割
   确保不会切在公式中间

3. 答案泄露检查
   check_answer_leakage(prefix, ground_truth)
   如果前缀中出现了 \boxed{GT} 或尾部出现 = GT
   → truncate_before_leakage() 截短

4. 组装 prompt
   [
     {"role": "system", "content": "Please reason step by step..."},
     {"role": "user", "content": question},
     {"role": "assistant", "content": prefix}  ← 关键！
   ]
```

### 为什么前缀放在 assistant 的 content 中？

verl/vLLM 的 chat template 处理：
```
<|im_start|>system
Please reason step by step...
<|im_end|>
<|im_start|>user
Find the integral of sin(x)...
<|im_end|>
<|im_start|>assistant
Let me use integration by parts...  ← 前缀
```

模型续写时从前缀末尾开始生成，response 只包含续写的部分。**不需要任何 response_mask 修改**。

---

## `batch_prepare_pg_prompts` — 批量准备

```python
def batch_prepare_pg_prompts(hard_zero_items, reference_answers, max_pg_samples=500):
```

在 `build_cycle_data.py` 中调用，为下一 cycle 的 parquet 准备 PG 记录。

生成的记录格式：
```python
{
    "data_source": "math",
    "prompt": [system, user, assistant_with_prefix],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": "..."},
    "extra_info": {
        "sample_id": "abc123...",
        "is_pg_sample": True,      # ← 标记为 PG 样本
        "prefix_ratio": 0.40,
        "prefix_len": 128,
    },
}
```

---

## PG 样本在训练中的行为

### GRPO 分组隔离
- PG 样本和 normal 样本有**不同的 prompt 内容**
- 在 batch 中获得不同的 UUID
- GRPO 按 UUID 分组时**自然分开**
- 不需要任何特殊处理

### LP 统计
- `is_pg_sample` 字段目前不影响 LP 更新
- 未来可扩展：PG 样本的 pass_rate 独立统计，不污染主 EMA

---

## 泄露检查的重要性

如果前缀中已经出现了最终答案，那模型续写后"正确"不代表它学会了推理。

```python
# 例子：GT = "42"
prefix = "...经过计算得到 x = 42，让我验证一下..."
# ← 答案已泄露！模型直接抄即可，没有学习价值

# 处理：截短到泄露位置之前
prefix = "...经过计算..."  # 截短后的安全前缀
```

---

## 面试加分点

- 能解释 "hard-zero 题为什么 LP weight 也帮不了"（因为 advantage=0）
- 能解释 "前缀放在 assistant content 中" 的工程巧妙性
- 能解释为什么选 40% 而不是 20%（GRPO 阶段模型已经比 warmstart 时更强，需要更短的提示）
- 能说明 PG 样本和 normal 样本的 GRPO 隔离是"天然"的
