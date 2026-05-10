# run_math_grpo.sh 修改详解

## 文件位置
`pipline/run_math_grpo.sh`

## 修改量
约 **20 行**新增代码

---

## 修改内容

### 1. Stage 1.5: LP 初始化（新增）

在原有 Stage 1（数据准备）之后、Stage 2（训练）之前，插入 LP 状态初始化：

```bash
# === LPPO: 初始化 LP State（从 student_responses 计算初始 P0）===
log "stage 1.5: 初始化 LPPO Learning Progress 状态"
python3 -m lppo.lp_init \
  --student_responses ${BASE_DIR}/sft_data/student_responses_120k_expanded_v4.jsonl \
  --output ${BASE_DIR}/checkpoints/$exp_name/lp_state.json \
  --beta 0.8 \
  --w_min 0.25 \
  --w_max 2.0
```

**作用**：用 DPO-v2 模型的历史采样结果初始化每道题的 P0。

### 2. Stage 2: GRPO 训练命令加 LPPO 参数

在 `verl.trainer.main_ppo` 命令末尾追加：

```bash
++lppo.enable=True \
++lppo.state_path=${BASE_DIR}/checkpoints/$exp_name/lp_state.json \
++lppo.module_path=${BASE_DIR} \
++lppo.ema_beta=0.8 \
++lppo.w_min=0.25 \
++lppo.w_max=2.0 \
++lppo.sigmoid_k=10 \
++lppo.tau=0.08 \
```

---

## Hydra 配置说明

verl 使用 OmegaConf/Hydra 管理配置。`++` 前缀表示"强制添加新 key"：

```
++lppo.enable=True
     │    │       │
     │    │       └── 值
     │    └── key path（在 config 中创建 lppo.enable）
     └── 强制添加（即使 config schema 中没有定义 lppo）
```

这些参数最终在 `ray_trainer.py` 中通过 `self.config.get("lppo", None)` 访问。

---

## 完整执行流程

```
run_math_grpo.sh 执行流程：

┌────────────────────────────────────────────────────┐
│ Stage 1: 数据准备                                   │
│                                                    │
│ prepare_math_rlvr_data.py                          │
│   输入: dpo_questions + student_responses + teacher │
│   输出: train.parquet (含 sample_id)               │
│         test.parquet                               │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│ Stage 1.5: LP 初始化 (新增)                         │
│                                                    │
│ lppo.lp_init                                       │
│   输入: student_responses_120k                     │
│   输出: lp_state.json (P0)                         │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│ Stage 2: GRPO 训练                                  │
│                                                    │
│ verl.trainer.main_ppo                              │
│   输入: train.parquet + lp_state.json              │
│   过程: 每步更新 LP state + weight × advantage     │
│   输出: model checkpoint + lp_state.json (更新后)  │
└────────────────────────────────────────────────────┘
```

---

## 环境变量与路径

```bash
# 关键路径
BASE_DIR=/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline
VERL_DIR=/cfs/cfs-esygraib/belvathliu/cv3/verl

# PYTHONPATH 设置（确保 lppo 包可 import）
export PYTHONPATH="${VERL_DIR}:${BASE_DIR}:${PYTHONPATH:-}"
#                    │           │
#                    │           └── lppo/ 和 reward_math_rlvr.py 所在目录
#                    └── verl 框架源码
```

`++lppo.module_path=${BASE_DIR}` 是备份机制：如果 PYTHONPATH 没设对，ray_trainer 会在 __init__ 中手动 `sys.path.insert(0, module_path)`。

---

## 如何关闭 LPPO

两种方式：

```bash
# 方式 1: 改配置参数
++lppo.enable=False

# 方式 2: 直接去掉所有 ++lppo.* 行
# ray_trainer.py 会走 self.config.get("lppo", None) → None → lppo_enabled=False
```

---

## 参数调优建议

| 场景 | 调整 |
|------|------|
| 权重分化太大（少数题主导梯度） | 降低 w_max 或增大 w_min |
| 权重几乎无区分（都在 1 附近） | 增大 k 或降低 tau |
| 已掌握题被过度惩罚 | 提高 w_min（如 0.4） |
| 学习效果不明显 | 降低 beta（更敏感地响应变化） |

---

## 面试加分点

- 能解释 `++` 前缀的 Hydra 语义
- 能说明为什么 LP init 在训练前执行（避免冷启动）
- 能说明 PYTHONPATH 设置确保跨目录 import 正确
- 能画出完整的 stage 依赖关系
