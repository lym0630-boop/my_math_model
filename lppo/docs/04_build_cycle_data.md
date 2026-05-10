# build_cycle_data.py 详解

## 文件定位
`pipline/lppo/build_cycle_data.py` — Cycle 重采样模块，训练中动态调整数据组成。

---

## 核心问题

### 静态数据的局限
标准 GRPO 训练中，每个 epoch 都遍历相同的数据：
- 已掌握的题反复出现 → 浪费 GPU 时间
- hard-zero 的题反复出现 → 没有学习信号
- 数据配比固定 → 无法适应模型能力变化

### Cycle 重采样的思路
在固定 step 间隔（一个 "cycle"）后：
1. 根据 LP state 重新评估每道题的学习价值
2. 增加"正在学习"的题的出现频率
3. 减少"已掌握"和"完全不会"的题
4. 为 hard-zero 题注入 PG 样本

---

## 数据配比设计

```python
DEFAULT_RATIOS = {
    'learning': 0.40,     # 正在学习 (lp > 0, 0.15 <= p <= 0.6)
    'sweet_spot': 0.25,   # 甜区 (0.1 <= p <= 0.5)
    'struggling': 0.15,   # 挣扎中 (0.05 <= p < 0.15)
    'hard_zero_pg': 0.10, # PG 样本 (p < 0.05，带前缀)
    'mastered': 0.05,     # 已掌握 (p > 0.8)
    'exploration': 0.05,  # 新题 (不在 LP state 中)
}
```

**面试解释**：
> 这个配比的设计逻辑是：
> - 65%（learning + sweet_spot）给最有学习价值的题
> - 15% 给挣扎中的题（给它们机会被"救"回来）
> - 10% 给 hard-zero（通过 PG 降低难度后尝试）
> - 5% 给已掌握的题（防遗忘）
> - 5% 给新题（保持 exploration，发现新的 sweet spot）

---

## 核心流程

```python
def build_cycle_data(lp_state_path, original_data_path, output_path, ...):
    # 1. 加载 LP state
    lp_mgr = LPStateManager()
    lp_mgr.load_state(lp_state_path)
    
    # 2. 获取题目分类
    categories = lp_mgr.get_problem_categories()
    # → {'hard_zero': [...], 'learning': [...], 'sweet_spot': [...], ...}
    
    # 3. 按配比采样
    for cat, ratio in ratios.items():
        n_target = int(target_size * ratio)
        # 从对应类别中随机采 n_target 题
        
    # 4. 为 hard-zero 生成 PG 样本
    pg_records = _generate_pg_samples(hard_zero_ids, ...)
    
    # 5. 保存为新的 parquet
    cycle_df.to_parquet(output_path)
```

---

## 与训练流程的集成

### 方案 A：外部调用（推荐，实现简单）
```bash
# run_math_grpo.sh 中
for cycle in 1 2 3; do
    # 运行一个 cycle 的训练
    python3 -m verl.trainer.main_ppo ... data.train_files=$cycle_data ...
    
    # 用 LP state 重建下一 cycle 数据
    python3 -m lppo.build_cycle_data \
        --lp_state_path checkpoints/$exp_name/lp_state.json \
        --original_data data/parquet_math_rlvr/train.parquet \
        --output data/parquet_math_rlvr/cycle_${next_cycle}.parquet
done
```

### 方案 B：内部触发（更自动化，需改 ray_trainer）
在 ray_trainer.py 中每隔 N steps 自动触发重采样。
→ 复杂度高，作为后续优化。

---

## `_generate_pg_samples()` 详解

```python
def _generate_pg_samples(hard_zero_ids, id_to_records, reference_answers_path, max_samples):
    # 1. 加载参考解
    ref_answers = {}
    with open(reference_answers_path) as f:
        for line in f:
            d = json.loads(line)
            ref_answers[d['question'][:120]] = d.get('answer', '')
    
    # 2. 对每道 hard-zero 题调用 prepare_pg_prompt
    for sid in hard_zero_ids:
        record = id_to_records[sid]
        question = _extract_question(record)
        result = prepare_pg_prompt(question, ref_answers[qk], gt)
        if result:
            # 修改记录的 prompt 为 PG 版本
            pg_record['prompt'] = result['prompt']
            pg_record['extra_info']['is_pg_sample'] = True
```

**关键设计**：PG 样本保留原始 record 的大部分字段（data_source, reward_model 等），
只替换 prompt 字段。这样 reward function 可以正常工作，不需要任何特殊处理。

---

## 面试追问预判

### Q: Cycle 多长合适？
**A**: 建议 300-500 steps（约一个 checkpoint 间隔）。太短会频繁重建数据（IO 开销），太长则失去自适应意义。在我们的配置中 save_freq=300，可以和 checkpoint 同步。

### Q: 重采样会不会导致数据泄露（train/test）？
**A**: 不会。重采样只从原始 train.parquet 中选取子集，test.parquet 始终不变。

### Q: 如果 learning 类别的题不够 40% 怎么办？
**A**: 代码有兜底逻辑——不足时从 sweet_spot 补充。实际中 sweet_spot 和 learning 有大量重叠（甜区且在进步的题），不太会出现不足。

### Q: exploration（新题）是什么？
**A**: 那些不在 LP state 中的题——可能是初始化时遗漏的，或后来加入题库的。给它们 5% 的名额确保不会永远被忽略。
