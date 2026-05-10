# prefix_guided_rollout.py 详解

## 文件定位
`pipline/lppo/prefix_guided_rollout.py` — 为 hard-zero 题目准备带前缀的 prompt。

---

## 核心问题

### hard-zero 题的困境
当一道题的 EMA pass rate p < 0.05（模型基本完全不会），GRPO 无法产生学习信号：
- 8 次 rollout 全部回答错误 → reward 全 0
- GRPO advantage = (reward - mean) / std = 0/0 → 没有梯度方向

### PG (Prefix-Guided) 解法
给模型一部分参考解作为"提示"：
- 截取参考解的前 40% 作为 prefix
- 模型从这个中间状态开始续写
- 更容易写出正确答案 → 产生正 reward → 有学习信号

**类比**：就像考试时老师给了部分解题步骤让学生续写，降低了任务难度。

---

## 复用关系

本模块复用 `prefix_guided_warmstart.py` 的 4 个核心函数：

| 函数 | 功能 | 原文件行号 |
|------|------|-----------|
| `find_semantic_breakpoints(text)` | 找语义断点（段落、公式结束处） | L134 |
| `cut_prefix(text, target_ratio, breakpoints)` | 按比例在断点处截取 | L148 |
| `check_answer_leakage(prefix, ground_truth)` | 检测前缀是否泄露答案 | L179 |
| `truncate_before_leakage(prefix, gt, breakpoints)` | 截短到泄露位置之前 | L202 |

**为什么不直接 import？**
实际代码中就是直接 import 的：
```python
from prefix_guided_warmstart import (
    find_semantic_breakpoints, cut_prefix,
    check_answer_leakage, truncate_before_leakage,
)
```

---

## 核心函数

### `prepare_pg_prompt(question, reference_answer, ground_truth)`

```python
def prepare_pg_prompt(question, reference_answer, ground_truth, prefix_ratio=0.40):
    # 1. 找语义断点
    breakpoints = find_semantic_breakpoints(reference_answer)
    
    # 2. 按 40% 比例截取
    prefix = cut_prefix(reference_answer, prefix_ratio, breakpoints)
    
    # 3. 答案泄露检查
    if check_answer_leakage(prefix, ground_truth):
        prefix = truncate_before_leakage(prefix, ground_truth, ...)
    
    # 4. 组装为 verl prompt 格式
    prompt = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefix},  # ← 关键：前缀在 assistant 位置
    ]
    return {"prompt": prompt, ...}
```

**面试解释**：
> PG prompt 的核心技巧是把前缀放在 assistant 的 content 里。verl 的 rollout 引擎看到非空的 assistant content 时，会把它作为已生成的部分，从后面开始续写。这样不需要修改任何 rollout 代码。

---

## 为什么选 40% 前缀比例

| 比例 | 优点 | 缺点 |
|------|------|------|
| 20% (短) | 模型需要自己推理更多，学习价值大 | 太难，可能还是答不对 |
| 40% (中) | 平衡难度和学习价值 | - |
| 60% (长) | 几乎肯定能答对 | 太容易，学不到东西 |

在训练中（而非 warmstart SFT），选择 40% 是因为：
- 目标不是"让模型续写正确"本身
- 目标是"产生一些正 reward，形成对比信号"
- 40% 前缀通常已经过了题目分析，到了关键计算步骤，正好让模型学"怎么从分析到计算"

---

## 答案泄露检查

### 为什么重要
如果前缀包含了 `\boxed{42}`（最终答案），模型只需要复制它就能得分。
这不是真正的学习，而是"作弊"。

### 检查逻辑
1. 检查 `\boxed{GT}` 是否在前缀中
2. 检查前缀最后 20% 中是否有 `= GT` 模式
3. 如果检测到泄露，在泄露位置之前的最近断点处截短

---

## 与 warmstart 的区别

| 维度 | prefix_guided_warmstart | prefix_guided_rollout |
|------|------------------------|----------------------|
| 时机 | 训练前（Stage 0.5） | 训练中（Cycle 间隔） |
| 目标 | 生成 SFT 样本 | 生成 RL 训练数据 |
| 前缀档次 | 3 档（短/中/长） | 1 档（中） |
| 每题采样次数 | 4 次（n=4） | 不采样，只准备 prompt |
| 质量过滤 | ROUGE-L + LCS 过滤 | 不需要（reward function 做判断） |

---

## 面试追问预判

### Q: PG 样本的 reward 怎么计算？和普通样本一样吗？
**A**: 完全一样。reward function 只看最终答案是否正确，不知道有没有前缀。对 PG 样本来说，模型只需要续写正确的后半部分就能得 1 分。

### Q: PG 样本会不会让模型"偷懒"（依赖前缀而不自己思考）？
**A**: 不会，原因有二：(1) PG 样本只占总数据的 ~10%，大部分训练还是标准 prompt；(2) PG 样本的 advantage 会被 LP weight 调节——一旦这些题的 pass rate 上升（从 hard-zero 离开），它们的 LP weight 和 PG 比例都会自动下降。

### Q: 如果参考解不存在怎么办？
**A**: 返回 None，这道题不生成 PG 样本。这种情况在实际中不常见，因为 dpo_questions 中绝大多数题都有完整参考解。
