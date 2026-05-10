# LPPO (Learning Progress PPO) 实现总索引

## 一、项目概述

**目标**：在标准 GRPO 训练中引入 Learning Progress (LP) 加权机制，让模型把训练精力集中在"正在学会"的题目上。

**核心洞察**：
- 已掌握的题（pass rate > 80%）：advantage ≈ 0，训练无意义
- 完全不会的题（pass rate ≈ 0%）：advantage ≈ 0，无学习信号
- 正在学习的题（pass rate 10%-50% 且在上升）：advantage 有方差，学习信号最强

LPPO 通过 LP weight 放大第三类题的 advantage，抑制前两类的无效训练。

---

## 二、架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    run_math_grpo.sh                          │
│  Stage 1: 数据准备 (prepare_math_rlvr_data.py + sample_id)  │
│  Stage 1.5: LP 初始化 (lp_init.py → lp_state_init.json)    │
│  Stage 2: GRPO 训练 (ray_trainer.py + LPPO hooks)           │
│  Stage 2.5: Cycle 重采样 (build_cycle_data.py) [可选]       │
│  Stage 3: 评估                                             │
└─────────────────────────────────────────────────────────────┘

训练循环内部：
┌──────────────────────────────────────────────────────────────┐
│  ray_trainer.py training loop                                │
│                                                              │
│  1. Rollout: 生成 responses                                  │
│  2. Reward: 计算 token_level_rewards                         │
│  3. compute_advantage(): 计算 GRPO advantage                 │
│  4. ★ Hook B: LP weight × advantage                         │
│  5. update_actor(): 用加权 advantage 更新模型                 │
│  6. ★ Hook D: checkpoint 保存时同步保存 LP state             │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、文件依赖顺序

实现和理解的顺序：

```
1. lp_state_manager.py    ← 核心，无外部依赖
   ↓
2. lp_init.py             ← 依赖 lp_state_manager
   ↓
3. prepare_math_rlvr_data.py (修改)  ← 加 sample_id 字段
   ↓
4. ray_trainer.py (修改)   ← 加 Hook B + Hook D，依赖 lp_state_manager
   ↓
5. prefix_guided_rollout.py ← 依赖 prefix_guided_warmstart 的函数
   ↓
6. build_cycle_data.py     ← 依赖 lp_state_manager + prefix_guided_rollout
   ↓
7. run_math_grpo.sh (修改) ← 串联所有模块
```

---

## 四、文件清单

### 新增文件

| 文件 | 功能 | 文档 |
|------|------|------|
| `lppo/__init__.py` | 模块入口 | - |
| `lppo/lp_state_manager.py` | LP 核心状态管理 | `docs/01_lp_state_manager.md` |
| `lppo/lp_init.py` | 从历史数据初始化 P0 | `docs/02_lp_init.md` |
| `lppo/prefix_guided_rollout.py` | PG 样本准备 | `docs/03_prefix_guided_rollout.md` |
| `lppo/build_cycle_data.py` | Cycle 重采样 | `docs/04_build_cycle_data.md` |

### 修改文件

| 文件 | 修改内容 | 文档 |
|------|----------|------|
| `prepare_math_rlvr_data.py` | +sample_id 到 extra_info | `docs/05_modify_prepare_data.md` |
| `verl/.../ray_trainer.py` | +Hook B, +Hook D | `docs/06_modify_ray_trainer.md` |
| `run_math_grpo.sh` | +LPPO 配置项 | `docs/07_modify_shell_script.md` |

---

## 五、LP 权重公式详解

### 5.1 EMA 更新
```
p_new = β × p_old + (1-β) × pass_rate
lp = p_new - p_old
```
- β = 0.8：当前 pass_rate 占 20% 权重
- 选 0.8 而非 0.9 是因为 GRPO 每题每 step 只采 8 次，需要较快响应

### 5.2 综合 Score
```
learnable = 4 × p × (1-p)       ← 在 p=0.5 时 = 1.0（最佳甜区）
score = 0.7 × max(lp, 0) + 0.3 × learnable
```
- 只取 lp 的正部分（不惩罚遗忘，由 KL loss 管）
- learnable 作为基线：即使 lp=0，甜区题也有较高 score

### 5.3 权重映射
```
weight = w_min + (w_max - w_min) × sigmoid(k × (score - τ))
```
- w_min=0.25, w_max=2.0, k=10, τ=0.08
- score < 0.08 → weight ≈ 0.25（已掌握/完全不会）
- score > 0.08 → weight ≈ 2.0（正在学习）
- sigmoid 保证过渡平滑

---

## 六、验证方法

### 6.1 单元测试
```bash
# 测试 LP State Manager
python3 -m lppo.lp_state_manager
# 期望输出：所有测试通过 ✅
```

### 6.2 集成测试
```bash
# 测试 LP 初始化
python3 -m lppo.lp_init --student_responses /path/to/student_responses.jsonl --output /tmp/test_lp_init.json
# 检查输出文件的 summary 字段
```

### 6.3 端到端验证
```bash
# 1. 准备数据（带 sample_id）
python3 prepare_math_rlvr_data.py --output_dir /tmp/test_lppo_data

# 2. 检查 sample_id 存在
python3 -c "import pandas as pd; df=pd.read_parquet('/tmp/test_lppo_data/train.parquet'); print(df.iloc[0]['extra_info'])"

# 3. 启动训练（观察 LP metrics）
# 在 tensorboard 中检查 lp/ 前缀的 metrics
```

### 6.4 面试演示点
- 展示 `lp_state_manager.py` 的 `_test_basic()` 输出，解释 weight 变化
- 用真实 `student_responses` 运行 `lp_init.py`，展示 P0 分布
- 对比有/无 LP 加权的 advantage 分布差异

---

## 七、面试话术要点

### Q: 为什么不直接用 pass_rate 作为权重？
**A**: pass_rate 是静态的难度指标，而 LP 捕捉的是"模型正在学什么"。一道 30% pass rate 的题如果 LP=0（没有进步），说明模型在这道题上停滞了，不值得加大训练强度。相反，一道从 5% 上升到 15% 的题，虽然绝对 pass_rate 低，但 LP > 0 表明模型正在突破，值得重点训练。

### Q: EMA 的 β=0.8 怎么选的？
**A**: GRPO rollout_n=8，每 step 一道题只有 8 个采样点。β=0.8 意味着每次新观测只占 20%，相当于 ~5 个 step 的滑动窗口，足以平滑单次采样的方差，又不至于响应太慢。如果 rollout_n 更大（比如 16），可以考虑用更小的 β。

### Q: 为什么 w_min=0.25 而不是 0？
**A**: 完全置零某些题的 advantage 会导致 policy gradient 的方向偏移——模型可能在已掌握的题上严重退化（因为没有维持信号）。w_min=0.25 保证即使"无用"的题也贡献 1/4 的梯度，配合 KL penalty 一起防止遗忘。

### Q: Cycle 重采样和 LP 权重的关系？
**A**: 两者互补。LP 权重是 step 级别的微调（在已有数据上改变梯度大小）；Cycle 重采样是 epoch 级别的粗调（改变模型看到的数据分布）。权重能做到"同一 batch 内差异化训练"，重采样能做到"不同 cycle 间数据换血"。
