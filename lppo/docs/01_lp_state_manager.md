# lp_state_manager.py 详解

## 文件定位
`pipline/lppo/lp_state_manager.py` — LPPO 的核心模块，所有其他模块都依赖它。

---

## 核心职责
1. **状态追踪**：为每道题维护 EMA pass rate (p) 和学习进度 (lp)
2. **权重计算**：根据 (p, lp) 计算 LP weight
3. **批量处理**：一次性处理整个 batch，返回与 advantage 对齐的权重
4. **持久化**：支持 JSON 序列化/反序列化

---

## 类设计

### `LPStateManager`

```python
class LPStateManager:
    def __init__(self, beta, w_min, w_max, k, tau, ...):
        self.states = {}  # sample_id → {p, lp, n_updates}
```

**为什么用 dict 而非 DataFrame？**
- 训练中频繁增删查改，dict 的 O(1) 查找比 DataFrame 快
- 状态数据不大（~15000 题），内存不是瓶颈
- JSON 序列化简单直接

---

## 核心方法逐行解析

### `update(sample_id, pass_rate)`

```python
def update(self, sample_id: str, pass_rate: float) -> dict:
    if sample_id not in self.states:
        # 首次遇到：直接用当前 pass_rate 初始化
        # 不用 EMA 是因为第一次没有历史可平滑
        self.states[sample_id] = {'p': pass_rate, 'lp': 0.0, 'n_updates': 1}
    else:
        state = self.states[sample_id]
        p_old = state['p']
        # EMA 更新公式：p_new = β*p_old + (1-β)*pass_rate
        p_new = self.beta * p_old + (1.0 - self.beta) * pass_rate
        lp = p_new - p_old  # 学习进度 = 新旧之差
        state['p'] = p_new
        state['lp'] = lp
        state['n_updates'] += 1
```

**面试解释**：
> EMA (Exponential Moving Average) 是一种加权平均方法。β=0.8 意味着新观测占 20%，历史占 80%。这样做的好处是：如果某次 rollout 只是运气好（8次里偶然答对了4次），EMA 不会让 p 跳太大。Learning Progress (lp) 就是 p 的变化量——正值表示模型在进步。

---

### `compute_weight(sample_id)`

```python
def compute_weight(self, sample_id: str) -> float:
    state = self.states[sample_id]
    p = state['p']
    lp = state['lp']
    
    # 1. learnable score: 数学函数 4*p*(1-p) 的形状是倒 U 型
    #    p=0 → 0, p=0.5 → 1.0, p=1 → 0
    learnable = 4.0 * p * (1.0 - p)
    
    # 2. 综合 score
    score = 0.7 * max(lp, 0.0) + 0.3 * learnable
    
    # 3. sigmoid 映射
    sigmoid_input = self.k * (score - self.tau)  # k=10, tau=0.08
    sigmoid_val = 1.0 / (1.0 + math.exp(-sigmoid_input))
    weight = self.w_min + (self.w_max - self.w_min) * sigmoid_val
    return weight
```

**面试解释**：
> 权重公式有三层含义：
> - `learnable`：从信息论角度，p=0.5 的题 entropy 最大，学习信号最丰富
> - `max(lp, 0)`：只奖励正向进步，负向进步（遗忘）由 KL loss 处理
> - `sigmoid`：把连续的 score 映射成有界权重，避免极端值

---

### `batch_update_and_get_weights(sample_ids, rewards, rollout_n)`

```python
def batch_update_and_get_weights(self, sample_ids, rewards, rollout_n=8):
    # Step 1: 按 sample_id 分组计算 pass_rate
    # 因为 GRPO 中同一题会出现 rollout_n 次
    # 例如：sample_ids = [q1, q1, q1, q1, q2, q2, q2, q2]
    #        rewards  = [ 0,  1,  0,  1,  1,  1,  1,  0]
    # → q1 的 pass_rate = 2/4 = 0.5
    # → q2 的 pass_rate = 3/4 = 0.75
    
    id_to_rewards = defaultdict(list)
    for sid, r in zip(sample_ids, rewards):
        id_to_rewards[sid].append(float(r))
    
    # Step 2: 更新 EMA
    for sid, reward_list in id_to_rewards.items():
        pass_rate = sum(1 for r in reward_list if r > 0) / len(reward_list)
        self.update(sid, pass_rate)
    
    # Step 3: 返回对齐的权重数组
    weights = np.array([self.compute_weight(sid) for sid in sample_ids])
    return weights
```

**面试解释**：
> 这个方法是 Hook B 的核心。GRPO 的 batch 中，每道题被重复 rollout_n 次。我们先按题分组统计 pass_rate，更新 EMA，然后给每个样本打上权重。同一题的所有 rollout 共享相同权重。

---

### `apply_lp_weights_to_advantages(advantages, sample_ids, rewards, rollout_n)`

```python
def apply_lp_weights_to_advantages(self, advantages, sample_ids, rewards, rollout_n=8):
    weights = self.batch_update_and_get_weights(sample_ids, rewards, rollout_n)
    weight_tensor = torch.tensor(weights, dtype=advantages.dtype, device=advantages.device)
    weight_tensor = weight_tensor.unsqueeze(-1)  # (batch,) → (batch, 1)
    weighted_advantages = advantages * weight_tensor  # 广播乘法
    return weighted_advantages
```

**面试解释**：
> advantages 的 shape 是 (batch_size, seq_len)，weight 的 shape 是 (batch_size,)。用 unsqueeze(-1) 加一维后，通过 PyTorch 的广播机制自动对每个 token 位置做逐元素乘法。这等价于对整个 response 的 advantage 统一缩放。

---

## 单元测试说明

文件末尾有 3 个测试函数：
1. `_test_basic()`: 验证 EMA 更新和权重计算的数值正确性
2. `_test_batch()`: 验证批量处理的 shape 和一致性
3. `_test_save_load()`: 验证序列化/反序列化

运行方式：
```bash
python3 -m lppo.lp_state_manager
```

---

## 面试追问预判

### Q: 如果题库很大（100k题），states dict 会不会爆内存？
**A**: 不会。每道题只存 3 个浮点数，100k 题约 2.4MB。

### Q: 多 GPU 训练时 LP state 会不会不一致？
**A**: LP state 只在 driver process 上维护（ray_trainer.py 的主进程），不存在多副本同步问题。compute_advantage 和 LP 加权都在 driver 上执行。

### Q: 为什么不用 numpy 直接向量化而是逐题循环？
**A**: 题目数量（~15000）远小于 batch_size × seq_len，瓶颈不在这里。且逐题更新逻辑更清晰、便于调试。
