# lp_state_manager.py 详解

## 文件位置
`pipline/lppo/lp_state_manager.py`

## 核心职责
LP State Manager 是 LPPO 系统的"大脑"，负责：
1. 维护每道题的 EMA pass rate
2. 计算 Learning Progress（学习进度）
3. 根据 LP 计算 advantage 权重
4. 序列化/反序列化状态

---

## 核心公式推导

### Step 1: EMA 更新
```
p_new = β × p_old + (1 - β) × pass_rate_current
```

**为什么用 EMA**：
- 单次 rollout 只有 8 个样本，方差极大
- EMA 用指数加权历史来平滑估计
- β=0.8 意味着：历史占 80%，当前观测占 20%
- 相当于约 5 次观测的滑动窗口

### Step 2: Learning Progress
```
lp = p_new - p_old
```

- `lp > 0`：模型正在学会这道题（最有价值的训练信号）
- `lp ≈ 0`：已经稳定了（可能已掌握或卡住）
- `lp < 0`：模型在遗忘（可能是其他题的 negative transfer）

### Step 3: Learnable Score
```
learnable = 4 × p × (1 - p)
```

这是二项分布方差的 4 倍：
- `p=0` 或 `p=1` 时 → learnable = 0（太难/太简单）
- `p=0.5` 时 → learnable = 1.0（最佳学习区间）
- 本质是 Bernoulli 熵的代理指标

### Step 4: 综合 Score
```
score = 0.7 × max(lp, 0) + 0.3 × learnable
```

- 只奖励**正向**学习进度（负 lp 视为 0）
- learnable 作为基线，确保"甜区"题始终有较高权重
- 7:3 的比例让"正在进步"比"处于甜区"更受重视

### Step 5: Sigmoid 映射
```
weight = w_min + (w_max - w_min) × σ(k × (score - τ))
```

其中 σ(x) = 1/(1+e^(-x))

- 将 score 平滑映射到 [w_min, w_max] = [0.25, 2.0]
- `k=10, τ=0.08`：score < 0.04 时 weight ≈ w_min，score > 0.12 时 weight ≈ w_max
- sigmoid 保证：(1) 权重在范围内；(2) 变化平滑，不会突变

---

## 代码逐段解析

### `__init__`（L39-73）
```python
def __init__(self, beta=0.8, w_min=0.25, w_max=2.0, k=10.0, tau=0.08, ...):
```
纯粹存储超参数 + 初始化空 dict。不做任何 IO。

### `update`（L75-108）
```python
def update(self, sample_id: str, pass_rate: float) -> dict:
```
- 首次遇到的题：直接用 pass_rate 初始化（不走 EMA）
- 之后的更新：EMA 公式 + 计算 lp

**设计选择**：首次用 pass_rate 直接赋值，而不是从 0 开始 EMA。原因是如果从 0 开始，第一次更新会产生巨大的正 lp，造成虚假的"大进步"。

### `compute_weight`（L110-149）
```python
def compute_weight(self, sample_id: str) -> float:
```
- 未知题返回 1.0（中性，不加权也不减权）
- 已知题按公式计算

### `batch_update_and_get_weights`（L151-187）— **Hook B 的核心入口**
```python
def batch_update_and_get_weights(self, sample_ids, rewards, rollout_n=8):
```

这是被 `ray_trainer.py` 直接调用的函数。流程：

1. **分组**：同一 sample_id 的所有 rollout 聚合
2. **计算 pass_rate**：`sum(r > 0) / count`
3. **更新 EMA**：调用 `self.update()`
4. **返回 weights**：对每个样本位置返回对应 weight

**关键细节**：GRPO 中同一题有 rollout_n 个样本（通过 `batch.repeat(n)`），它们共享同一个 sample_id。这里先聚合后计算，确保同一题的所有 rollout 获得相同权重。

### `save_state` / `load_state`（L252-292）
纯 JSON 序列化。保存时额外包含 config 和 summary，方便人工检查。

### `apply_lp_weights_to_advantages`（L294-334）— 备选入口
封装了完整的"更新+加权"逻辑，但 ray_trainer 中使用了更灵活的分步调用。

---

## 数据流示意

```
ray_trainer.py 调用链：

batch.batch["token_level_scores"]  ──→  sum(dim=-1)  ──→  rewards (bs,)
batch.non_tensor_batch["extra_info"]  ──→  提取 sample_id  ──→  sample_ids (bs,)
                                                                    │
                                    ┌───────────────────────────────┘
                                    ▼
              batch_update_and_get_weights(sample_ids, rewards)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
               分组聚合      update(每题)     compute_weight
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                           weights (bs,) → normalize → × advantage
```

---

## 单元测试解读

文件底部有 3 个测试：
1. `_test_basic`：验证 EMA 数值正确 + 权重方向合理
2. `_test_batch`：验证批量处理 + 同题同权重
3. `_test_save_load`：验证序列化 round-trip

运行：`python3 -m lppo.lp_state_manager`

---

## 面试加分点

- **EMA 系数选择**：可以解释 β=0.8 对应约 5 次有效观测窗口 (1/(1-β)=5)
- **Sigmoid vs 线性映射**：sigmoid 保证权重不会突变，训练更稳定
- **batch mean normalization**：在 ray_trainer 中做了 `weights / weights.mean()`，确保加权后总梯度量级不变
