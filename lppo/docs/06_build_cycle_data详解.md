# build_cycle_data.py 详解

## 文件位置
`pipline/lppo/build_cycle_data.py`

## 核心职责
在每个 training cycle 结束后，根据 LP state 重新构造下一 cycle 的训练数据，实现动态数据配比调整。

---

## 动机：为什么需要 Cycle 重采样？

标准 GRPO 对所有题均匀采样，但训练过程中：

| 题目状态 | pass rate | GRPO advantage | 训练价值 |
|----------|-----------|----------------|----------|
| 已掌握 | ≈ 1.0 | ≈ 0（全对，无对比） | 极低 |
| 完全不会 | ≈ 0.0 | ≈ 0（全错，无对比） | 极低 |
| 正在学习 | 0.1~0.8 | 有正有负，信息量大 | **很高** |

**LP weight 是 token-level 的调整**（改变梯度权重），**Cycle 重采样是 data-level 的调整**（改变题目出现频率）。两者互补：

- LP weight：精细调节每道题的梯度贡献
- Cycle 重采样：粗粒度调节哪些题出现在 batch 中

---

## 数据配比策略

```python
DEFAULT_RATIOS = {
    'learning': 0.40,     # 正在学习的题（核心目标）
    'sweet_spot': 0.25,   # 甜区题（稳定训练信号）
    'struggling': 0.15,   # 挣扎中的题（给它们机会）
    'hard_zero_pg': 0.10, # hard-zero 的 PG 样本
    'mastered': 0.05,     # 已掌握（防遗忘）
    'exploration': 0.05,  # 未见过的新题
}
```

### 各类别定义

| 类别 | LP 条件 | 说明 |
|------|---------|------|
| learning | p∈[0.15, 0.6], lp > 0 | 模型正在进步的题 |
| sweet_spot | p∈[0.1, 0.5] | 最佳难度区间 |
| struggling | p∈[0.05, 0.15] | 有一点点会但很挣扎 |
| hard_zero | p < 0.05 | 完全不会 |
| mastered | p > 0.8 | 已经掌握 |
| exploration | 不在 LP state 中 | 从未见过的题 |

---

## 核心函数 `build_cycle_data`

```python
def build_cycle_data(
    lp_state_path,           # LP state JSON
    original_data_path,      # 原始 train.parquet
    output_path,             # 输出的新 cycle parquet
    reference_answers_path,  # 参考解（PG 用）
    target_size=None,        # 目标数量
    ratios=None,             # 配比
    seed=42,
)
```

### 处理流程

```
1. 加载 LP state
   LPStateManager.load_state(lp_state_path)

2. 加载原始数据
   pd.read_parquet(original_data_path)
   构建 sample_id → record 映射

3. 获取题目分类
   lp_mgr.get_problem_categories()
   → {'learning': [...], 'sweet_spot': [...], 'mastered': [...], ...}

4. 按配比采样
   对每个类别：
     n_target = total × ratio
     从该类别的 sample_ids 中随机选 n_target 个
     找到对应的 parquet record

5. 生成 PG 样本
   对 hard_zero 类别：
     调用 prefix_guided_rollout.prepare_pg_prompt()
     生成带前缀的新记录

6. 补足
   如果数量不够，从 sweet_spot + learning 补充

7. 保存
   shuffle + to_parquet
```

---

## 与 run_math_grpo.sh 的配合

### 单 Cycle 模式（当前）
```bash
# Stage 1: 数据准备（一次性）
python3 prepare_math_rlvr_data.py → train.parquet

# Stage 1.5: LP 初始化
python3 -m lppo.lp_init → lp_state.json

# Stage 2: 训练（使用固定的 train.parquet）
python3 -m verl.trainer.main_ppo ...
```

### 多 Cycle 模式（进阶）
```bash
for cycle in 1 2 3; do
  # 基于上一 cycle 的 LP state 重新采样
  python3 -m lppo.build_cycle_data \
    --lp_state_path checkpoints/exp/lp_state.json \
    --original_data data/train_full.parquet \
    --output data/cycle_${cycle}/train.parquet \
    --reference_answers sft_data/dpo_questions_120k_expanded_v4.jsonl

  # 用新数据训练
  python3 -m verl.trainer.main_ppo \
    data.train_files=data/cycle_${cycle}/train.parquet \
    ...
done
```

---

## 辅助函数

### `_get_sample_id(row)` — 从 DataFrame row 提取 sample_id
```python
def _get_sample_id(row) -> str:
    extra = row.get('extra_info', {})
    if isinstance(extra, str):
        extra = json.loads(extra)
    return extra.get('sample_id', '')
```

**注意**：parquet 中 extra_info 可能是 dict 也可能是 JSON string，需要兼容。

### `_generate_pg_samples(...)` — PG 样本生成
调用 `prefix_guided_rollout.prepare_pg_prompt()` 并组装完整记录。

### `_extract_question(record)` / `_extract_ground_truth(record)`
从 verl parquet 记录中提取问题和答案（需要解析 prompt list 格式）。

---

## 面试加分点

- 能区分 "token-level weight"（LP weight）和 "data-level weight"（cycle 重采样）
- 能解释 5% mastered 题的保留是为了"防遗忘"（catastrophic forgetting）
- 能解释 5% exploration 是为了让 LP state 覆盖更多题目
- 能说明多 cycle 训练的收敛优势：每个 cycle 的数据更有针对性
