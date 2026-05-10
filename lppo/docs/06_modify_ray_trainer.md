# ray_trainer.py 修改详解

## 修改目标
在 verl 的核心训练循环中添加 2 个 LPPO hook：
- **Hook B**: `compute_advantage()` 之后，用 LP weight 乘以 advantage
- **Hook D**: `_save_checkpoint()` 时保存 LP state

---

## 文件位置
`verl/verl/trainer/ppo/ray_trainer.py` (1393 行)

---

## 修改点总览

| Hook | 位置 | 行号(原) | 功能 |
|------|------|----------|------|
| 初始化 | `__init__` | L293 | 创建 LPStateManager，加载 P0 |
| Hook B | `compute_advantage()` 之后 | L1280-1288 | LP weight × advantage |
| Hook D | `_save_checkpoint()` 内 | L914-950 | 保存 LP state JSON |
| Metrics | metrics 收集处 | L1359-1370 | 记录 LP 统计到 tensorboard |

---

## 修改 1：`__init__` 中初始化 LP State Manager

### 位置：L293 `def __init__(...)` 方法末尾（约 L372 之后）

```python
# ============ LPPO: 初始化 LP State Manager ============
# 检查是否启用 LPPO
self.use_lppo = config.get("lppo", {}).get("enable", False)
if self.use_lppo:
    from lppo.lp_state_manager import LPStateManager
    from lppo.lp_init import init_lp_state_from_student_responses

    lppo_config = config.lppo
    self.lp_manager = LPStateManager(
        beta=lppo_config.get("beta", 0.8),
        w_min=lppo_config.get("w_min", 0.25),
        w_max=lppo_config.get("w_max", 2.0),
        k=lppo_config.get("k", 10.0),
        tau=lppo_config.get("tau", 0.08),
    )
    
    # 尝试从 checkpoint 恢复 LP state
    lp_state_path = os.path.join(
        config.trainer.default_local_dir, "lp_state.json"
    )
    if not self.lp_manager.load_state(lp_state_path):
        # checkpoint 中无 LP state → 从 student_responses 初始化 P0
        student_resp_path = lppo_config.get("student_responses_path", "")
        if student_resp_path and os.path.exists(student_resp_path):
            init_lp_state_from_student_responses(
                student_resp_path, self.lp_manager
            )
        else:
            print("[LPPO] 无 student_responses，LP state 从空开始")
    
    print(f"[LPPO] 已启用，管理 {len(self.lp_manager.states)} 道题")
# ============ END LPPO Init ============
```

**面试解释**：
> 初始化逻辑有优先级：先尝试从 checkpoint 恢复（训练中断后继续），如果没有再从 student_responses 初始化 P0（首次训练）。这样无论是冷启动还是断点续训都能正确工作。

---

## 修改 2：Hook B — LP Weight × Advantage

### 位置：L1280-1288 `compute_advantage()` 调用之后

```python
# 原始代码 L1280-1288
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam,
    num_repeat=self.config.actor_rollout_ref.rollout.n,
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    config=self.config.algorithm,
)

# ============ LPPO Hook B: LP Weight × Advantage ============
if self.use_lppo:
    # 获取 sample_ids：从 extra_info 中提取
    # batch.non_tensor_batch 中的 uid 是 GRPO 用的 group id
    # sample_id 需要从 extra_info 中获取
    sample_ids = self._get_sample_ids_from_batch(batch)
    
    # 获取 rewards：token_level_scores 求和得到每样本总 reward
    rewards = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
    
    # 用 LP Manager 加权 advantage
    batch.batch["advantages"] = self.lp_manager.apply_lp_weights_to_advantages(
        advantages=batch.batch["advantages"],
        sample_ids=sample_ids,
        rewards=rewards,
        rollout_n=self.config.actor_rollout_ref.rollout.n,
    )
    
    # 记录 LP metrics
    lp_summary = self.lp_manager.get_state_summary()
    metrics.update({
        "lp/avg_p": lp_summary['avg_p'],
        "lp/avg_lp": lp_summary['avg_lp'],
        "lp/avg_weight": lp_summary['avg_weight'],
        "lp/hard_zero_count": lp_summary['hard_zero_count'],
        "lp/sweet_spot_count": lp_summary['sweet_spot_count'],
    })
# ============ END LPPO Hook B ============
```

### 辅助方法：`_get_sample_ids_from_batch()`

```python
def _get_sample_ids_from_batch(self, batch) -> np.ndarray:
    """从 batch 的 non_tensor_batch 中提取 sample_id
    
    verl 的数据流：
      parquet 中 extra_info.sample_id
      → DataLoader 加载后在 batch.non_tensor_batch["extra_info"] 中
      → 需要解析出来
    """
    extra_infos = batch.non_tensor_batch.get("extra_info", None)
    if extra_infos is None:
        # 降级：用 uid（每 step 随机分配的 group id）
        return batch.non_tensor_batch["uid"]
    
    sample_ids = []
    for info in extra_infos:
        if isinstance(info, dict):
            sample_ids.append(info.get("sample_id", "unknown"))
        elif isinstance(info, str):
            import json
            d = json.loads(info)
            sample_ids.append(d.get("sample_id", "unknown"))
        else:
            sample_ids.append("unknown")
    
    return np.array(sample_ids, dtype=object)
```

---

## 修改 3：Hook D — Checkpoint 保存 LP State

### 位置：L914 `_save_checkpoint()` 方法内，actor checkpoint 保存之后

```python
def _save_checkpoint(self):
    from verl.utils.fs import local_mkdir_safe

    local_global_step_folder = os.path.join(
        self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
    )
    # ... 原始 actor/critic checkpoint 保存代码 ...
    
    self.actor_rollout_wg.save_checkpoint(
        actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
    )

    # ============ LPPO Hook D: 保存 LP State ============
    if self.use_lppo:
        lp_state_path = os.path.join(local_global_step_folder, "lp_state.json")
        self.lp_manager.save_state(lp_state_path)
        # 同时保存一份到根目录（方便 resume 时快速找到最新状态）
        lp_state_latest = os.path.join(
            self.config.trainer.default_local_dir, "lp_state.json"
        )
        self.lp_manager.save_state(lp_state_latest)
        print(f"[LPPO] LP state 已保存: {lp_state_path}")
    # ============ END LPPO Hook D ============
    
    if self.use_critic:
        # ... 原始 critic 保存代码 ...
```

**设计决策**：LP state 同时保存两份：
1. `global_step_N/lp_state.json` — 随 checkpoint 版本化，便于回滚
2. `lp_state.json` — 根目录最新版，resume 时直接加载

---

## 完整修改 Diff 预览

```diff
--- a/verl/verl/trainer/ppo/ray_trainer.py
+++ b/verl/verl/trainer/ppo/ray_trainer.py
@@ -370,6 +370,28 @@ class RayPPOTrainer:
         else:
             raise NotImplementedError
 
+        # ============ LPPO Init ============
+        self.use_lppo = config.get("lppo", {}).get("enable", False)
+        if self.use_lppo:
+            from lppo.lp_state_manager import LPStateManager
+            from lppo.lp_init import init_lp_state_from_student_responses
+            lppo_config = config.lppo
+            self.lp_manager = LPStateManager(
+                beta=lppo_config.get("beta", 0.8),
+                ...
+            )
+            ...
+        # ============ END LPPO Init ============
+
@@ -1288,6 +1310,20 @@ class RayPPOTrainer:
                         config=self.config.algorithm,
                     )
 
+                    # ============ LPPO Hook B ============
+                    if self.use_lppo:
+                        sample_ids = self._get_sample_ids_from_batch(batch)
+                        rewards = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
+                        batch.batch["advantages"] = self.lp_manager.apply_lp_weights_to_advantages(
+                            advantages=batch.batch["advantages"],
+                            sample_ids=sample_ids,
+                            rewards=rewards,
+                            rollout_n=self.config.actor_rollout_ref.rollout.n,
+                        )
+                        ...
+                    # ============ END LPPO Hook B ============
+
@@ -944,6 +966,12 @@ class RayPPOTrainer:
         self.actor_rollout_wg.save_checkpoint(...)
 
+        # ============ LPPO Hook D ============
+        if self.use_lppo:
+            lp_state_path = os.path.join(local_global_step_folder, "lp_state.json")
+            self.lp_manager.save_state(lp_state_path)
+            ...
+        # ============ END LPPO Hook D ============
+
         if self.use_critic:
```

---

## 关键设计决策

### 为什么在 compute_advantage 之后而非之前加权？
- GRPO 的 advantage 需要在 group 内做 normalization（减均值除标准差）
- 如果先加权再 normalize，权重效果会被 normalize 消除
- 先 normalize 再加权，才能真正改变不同题目的梯度贡献比例

### 为什么不修改 compute_advantage 函数本身？
- `compute_advantage()` 是通用函数，被 GAE/GRPO/REINFORCE++ 等多个算法共享
- 修改它会影响所有算法
- Hook 模式（外部加权）更安全，且可以通过 `use_lppo` flag 开关

### LP metrics 记录了什么？
| Metric | 含义 | 期望趋势 |
|--------|------|----------|
| `lp/avg_p` | 全局平均 pass rate | 缓慢上升 |
| `lp/avg_lp` | 平均学习进度 | 初期正值，后期趋零 |
| `lp/avg_weight` | 平均 LP weight | 初期 >1，后期趋近 1 |
| `lp/hard_zero_count` | 完全不会的题数 | 下降 |
| `lp/sweet_spot_count` | 甜区题数 | 先升后降 |

---

## 面试追问预判

### Q: 这个 hook 会增加多少计算开销？
**A**: 极少。LP 权重计算只涉及 ~15000 次浮点运算（每题一次 sigmoid），在 GPU 训练的时间尺度上完全可以忽略（< 1ms vs 整个 step 数十秒）。

### Q: 如果 batch 中某些样本没有 sample_id 怎么办？
**A**: `_get_sample_ids_from_batch()` 有降级逻辑——如果 extra_info 中没有 sample_id，返回 "unknown"，对应权重为 1.0（中性，不加权）。

### Q: compute_advantage 后 advantage 已经 normalized 了，再乘权重会不会破坏分布？
**A**: 会改变分布，但这正是我们的目的。加权后，learning 题的 advantage 被放大，mastered 题的被缩小。policy gradient 的方向因此偏向 learning 题。整体 loss scale 可能需要调整 lr，但实验中 w_max=2.0 的范围不需要改 lr。
