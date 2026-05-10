# run_math_grpo.sh 修改详解

## 修改目标
在训练脚本中添加 LPPO 配置参数，启用 LP 加权机制。

---

## 文件位置
`pipline/run_math_grpo.sh` (158 行)

---

## 修改内容

### 1. 新增 Stage 1.5：LP 初始化

在 Stage 1 和 Stage 2 之间添加：

```bash
# ===== Stage 1.5: LPPO 初始化 (从 student_responses 初始化 P0) =====
if [ $stage -le 1 ] && [ $stop_stage -ge 2 ]; then
  log "stage 1.5: 初始化 LP State (P0)"
  
  # 从 student_responses 计算每题的初始 pass rate
  python3 -m lppo.lp_init \
    --student_responses ${BASE_DIR}/sft_data/student_responses_120k_expanded_v4.jsonl \
    --output ${BASE_DIR}/checkpoints/$exp_name/lp_state.json \
    --beta 0.8 \
    --w_min 0.25 \
    --w_max 2.0
fi
```

### 2. Stage 2 训练命令中添加 LPPO 配置

在 `python3 -m verl.trainer.main_ppo \` 命令中追加：

```bash
      # ... 原有配置 ...
      trainer.val_before_train=True \
      ++lppo.enable=True \
      ++lppo.beta=0.8 \
      ++lppo.w_min=0.25 \
      ++lppo.w_max=2.0 \
      ++lppo.k=10.0 \
      ++lppo.tau=0.08 \
      ++lppo.student_responses_path=${BASE_DIR}/sft_data/student_responses_120k_expanded_v4.jsonl \
      2>&1 | tee $LOG_FILE
```

---

## 完整修改 Diff

```diff
--- a/pipline/run_math_grpo.sh
+++ b/pipline/run_math_grpo.sh
@@ -49,6 +49,17 @@ if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
     --seed 42
 fi
 
+# ===== Stage 1.5: LPPO 初始化 =====
+if [ $stage -le 1 ] && [ $stop_stage -ge 2 ]; then
+  log "stage 1.5: 初始化 LP State (P0)"
+  python3 -m lppo.lp_init \
+    --student_responses ${BASE_DIR}/sft_data/student_responses_120k_expanded_v4.jsonl \
+    --output ${BASE_DIR}/checkpoints/$exp_name/lp_state.json \
+    --beta 0.8 \
+    --w_min 0.25 \
+    --w_max 2.0
+fi
+
 # ===== Stage 2: GRPO 训练 =====
 if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
@@ -146,6 +157,13 @@ if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
       trainer.resume_mode='auto' \
       trainer.default_local_dir=${BASE_DIR}/checkpoints/$exp_name \
       trainer.total_epochs=1 \
       trainer.val_before_train=True \
+      ++lppo.enable=True \
+      ++lppo.beta=0.8 \
+      ++lppo.w_min=0.25 \
+      ++lppo.w_max=2.0 \
+      ++lppo.k=10.0 \
+      ++lppo.tau=0.08 \
+      ++lppo.student_responses_path=${BASE_DIR}/sft_data/student_responses_120k_expanded_v4.jsonl \
       2>&1 | tee $LOG_FILE
 fi
```

---

## 配置参数说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `lppo.enable` | True | 启用/禁用 LPPO（方便 A/B 测试） |
| `lppo.beta` | 0.8 | EMA 衰减系数 |
| `lppo.w_min` | 0.25 | 最小权重 |
| `lppo.w_max` | 2.0 | 最大权重 |
| `lppo.k` | 10.0 | sigmoid 陡峭度 |
| `lppo.tau` | 0.08 | sigmoid 中心阈值 |
| `lppo.student_responses_path` | ... | P0 初始化数据路径 |

---

## `++` 前缀语法说明

verl 使用 hydra/OmegaConf 做配置管理：
- `key=value`：覆盖已有配置项
- `+key=value`：添加新配置项（配置文件中没有的）
- `++key=value`：如果存在则覆盖，不存在则创建

我们用 `++lppo.*` 是因为 verl 默认配置中没有 `lppo` 这个 group，
`++` 确保无论有无默认值都能正确设置。

---

## PYTHONPATH 配置

确保 lppo 模块可被 import：

```bash
# 原有的 PYTHONPATH 设置（L77）
export PYTHONPATH="${VERL_DIR}:${BASE_DIR}:${PYTHONPATH:-}"
```

`${BASE_DIR}` 是 `pipline/` 目录，而 lppo 模块在 `pipline/lppo/` 下，
所以 `from lppo.lp_state_manager import ...` 可以正常工作。

---

## 可选：Stage 2.5 Cycle 重采样

如果要启用 Cycle 重采样，在 Stage 2 和 Stage 3 之间添加循环逻辑：

```bash
# ===== Stage 2.5: Cycle 重采样（可选进阶功能）=====
# 需要修改 Stage 2 为循环模式
NUM_CYCLES=3
CYCLE_STEPS=500

for cycle in $(seq 1 $NUM_CYCLES); do
  log "Cycle $cycle / $NUM_CYCLES"
  
  if [ $cycle -gt 1 ]; then
    # 根据 LP state 重建训练数据
    python3 -m lppo.build_cycle_data \
      --lp_state_path ${BASE_DIR}/checkpoints/$exp_name/lp_state.json \
      --original_data $data_dir/train.parquet \
      --output $data_dir/cycle_${cycle}.parquet \
      --reference_answers ${BASE_DIR}/sft_data/dpo_questions_120k_expanded_v4.jsonl \
      --seed $((42 + cycle))
    
    cycle_data=$data_dir/cycle_${cycle}.parquet
  else
    cycle_data=$data_dir/train.parquet
  fi
  
  # 运行训练（使用当前 cycle 的数据）
  python3 -m verl.trainer.main_ppo \
    ... \
    data.train_files=$cycle_data \
    trainer.total_training_steps=$CYCLE_STEPS \
    ...
done
```

**注意**：Cycle 重采样是进阶功能，可以先不实现。核心的 Hook B + Hook D 已经提供了主要的 LP 加权效果。

---

## 面试话术

> shell 脚本的修改很简单——就是通过 hydra 的 `++` 语法把 LPPO 的超参数传入训练框架。关键在于设计的灵活性：`lppo.enable=True/False` 一个开关就能对比有无 LP 加权的实验结果。这是做 RL 研究的标准实践——所有 ablation 都应该只需要改配置，不需要改代码。

---

## 验证方法

```bash
# 1. 检查配置是否正确传入
python3 -m verl.trainer.main_ppo \
    ++lppo.enable=True \
    ++lppo.beta=0.8 \
    --cfg job  # 打印完整配置，确认 lppo 节点存在

# 2. 实际运行验证（看日志）
# 期望看到:
#   [LPPO] 已启用，管理 15000 道题
#   [LPPO] LP state 已保存: .../global_step_300/lp_state.json

# 3. 检查 tensorboard
# 期望看到 lp/ 前缀的 metrics 曲线
```
