# lp_init.py 详解

## 文件位置
`pipline/lppo/lp_init.py`

## 核心职责
从 `student_responses_*.jsonl` 中提取历史 pass rate，初始化 LP State Manager 的状态（P0）。

---

## 为什么需要初始化？

如果不初始化（所有题从 p=0 开始）：

```
Step 1: 题目 A 首次被采样，pass_rate = 0.3
  → p = 0.3 (首次直接赋值)
  → lp = 0 (首次无历史)

Step 2: 题目 A 第二次被采样，pass_rate = 0.4
  → p_new = 0.8 × 0.3 + 0.2 × 0.4 = 0.32
  → lp = 0.32 - 0.3 = 0.02 (真实的学习进度)
```

但如果 **有初始化**：
```
P0 阶段：从 student_responses 得知 题目 A pass_rate = 0.3
  → 直接设 p = 0.3

Step 1: 题目 A 被采样，pass_rate = 0.4
  → p_new = 0.8 × 0.3 + 0.2 × 0.4 = 0.32
  → lp = 0.32 - 0.3 = 0.02 ← 从第一步就能正确计算 LP！
```

**没有初始化的风险**：训练前几个 step 所有首次出现的题都会有 lp=0（因为没有历史），LP 加权机制要到第二次遇到这道题才生效。初始化让系统从第一步就"热启动"。

---

## 数据来源

`student_responses_120k_expanded_v4.jsonl` 格式：
```json
{
  "question": "Find the value of x if 3x + 5 = 20",
  "ground_truth": "5",
  "num_correct": 3,
  "num_total": 8,
  "responses": [...],
  ...
}
```

关键字段：
- `num_correct / num_total` = 该题在 DPO-v2 模型上的 pass rate
- 这是在 SFT + DPO 之后、GRPO 之前的模型能力

---

## `compute_sample_id` 函数

```python
def compute_sample_id(question: str, ground_truth: str = "") -> str:
    key = question.strip() + "||" + ground_truth.strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]
```

**与 `prepare_math_rlvr_data.py` 中的 `make_sample_id` 完全一致**。

设计考虑：
- 用 `question + ground_truth` 而非单独 question：同一题可能有不同 GT 版本
- SHA1 前 16 位 = 64 bit 空间：120K 题库碰撞概率 < 10^-8
- `strip()` 确保空格不影响 hash

---

## 核心函数 `init_lp_state_from_student_responses`

```python
def init_lp_state_from_student_responses(
    student_responses_path, lp_manager=None, output_path=None
) -> LPStateManager:
```

逻辑很简单：
1. 遍历 jsonl 文件
2. 对每道题算 `sample_id` 和 `pass_rate`
3. 直接设置 `states[sample_id] = {'p': pass_rate, 'lp': 0.0, 'n_updates': 1}`

**注意**：这里是**直接赋值**，不走 EMA 更新。因为：
- 初始化时没有"上一次"可以计算 lp
- 如果走 EMA（从 0 开始），会产生虚假的学习进度

---

## 命令行使用

```bash
python3 -m lppo.lp_init \
  --student_responses sft_data/student_responses_120k_expanded_v4.jsonl \
  --output checkpoints/exp_name/lp_state.json \
  --beta 0.8 \
  --w_min 0.25 \
  --w_max 2.0
```

输出示例：
```
[LP Init] 从 student_responses 初始化 118432 道题的 P0
[LP Init] 平均 P0: 0.342
[LP Init] hard_zero (p<0.05): 8234
[LP Init] sweet_spot (0.1~0.5): 52341
[LP Init] mastered (p>0.8): 12897
[LP Init] 已保存初始状态到: checkpoints/.../lp_state.json
```

---

## 与训练流程的关系

```
run_math_grpo.sh 中的调用顺序：

Stage 1: prepare_math_rlvr_data.py → train.parquet (含 sample_id)
Stage 1.5: lp_init.py → lp_state.json (P0 初始化)
Stage 2: verl 训练 → ray_trainer.py 加载 lp_state.json, 在线更新
```

---

## 面试加分点

- 能解释为什么"冷启动"是问题以及如何解决
- 能说明 sample_id 一致性的重要性（如果 init 和 training 的 hash 不一致就全白做了）
- 能说明 P0 的数值含义："这是 GRPO 训练之前模型的能力基线"
