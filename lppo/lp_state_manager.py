"""
LP State Manager — LPPO 核心状态管理

功能：
  1. 维护每道题的 EMA pass rate (p) 和学习进度 (lp)
  2. 根据 rollout 结果更新状态
  3. 计算 LP weight 用于加权 advantage
  4. 支持序列化/反序列化（checkpoint 保存/恢复）

设计原理：
  - EMA (Exponential Moving Average) 平滑 pass rate，避免单次采样噪声
  - Learning Progress = p_new - p_old，正值表示模型正在学会这道题
  - Learnable score = 4*p*(1-p)，在 p=0.5 时最大，表示题目难度适中
  - 最终 weight 通过 sigmoid 映射到 [w_min, w_max] 区间

与 ray_trainer.py 的交互：
  - Hook B: compute_advantage 后调用 apply_lp_weights() 加权
  - Hook D: _save_checkpoint 时调用 save_state() 持久化
"""

import json
import math
import os
import numpy as np
from collections import defaultdict
from typing import Optional


class LPStateManager:
    """
    学习进度状态管理器
    
    每道题维护：
      - p: EMA pass rate（指数移动平均通过率）
      - lp: learning progress（本次更新的学习进度 = p_new - p_old）
      - n_updates: 该题被更新的次数
    """

    def __init__(
        self,
        beta: float = 0.8,       # EMA 衰减系数，越大越平滑
        kappa: float = 8.0,      # sigmoid 陡峭度（论文 κ=8.0）
        b: float = 0.5,          # sigmoid 偏置（论文 b=0.5）
    ):
        """
        初始化 LP State Manager

        参数说明（对齐 LPPO 论文）：
          beta: EMA 系数。0.8 意味着当前 pass_rate 占 20%，历史占 80%。
                选择 0.8 是因为 GRPO 通常每题一个 step 只采 8 次，
                需要足够平滑来对抗采样方差。
          kappa: sigmoid 的陡峭度。κ=8.0 时：
                 Δ=±0.1 → weight ≈ 1.0 ± 0.19
                 Δ=±0.5 → weight ≈ 1.0 ± 0.48
          b: sigmoid 偏置。b=0.5 确保 weight 范围为 [0.5, 1.5]，
             中心在 Δ=0 时 weight=1.0（中性）。

        权重公式（论文原版）：
          w_i(t) = sigmoid(κ * Δ_i(t)) + b
          其中 Δ_i(t) = p_ema_new - p_ema_old（学习进度）

        权重范围：
          Δ → -∞: weight → 0 + b = 0.5（大幅退步，缩小梯度）
          Δ = 0:  weight = 0.5 + 0.5 = 1.0（无变化，中性）
          Δ → +∞: weight → 1 + b = 1.5（大幅进步，放大梯度）
        """
        # 超参数
        self.beta = beta
        self.kappa = kappa
        self.b = b

        # 状态存储：sample_id → {p, lp, n_updates}
        self.states = {}

    def update(self, sample_id: str, pass_rate: float) -> dict:
        """
        用本轮 rollout 的 pass_rate 更新某道题的状态
        
        Args:
            sample_id: 题目唯一标识（对应 parquet 中的 sample_id）
            pass_rate: 本轮 rollout 中该题的正确率 (0.0~1.0)
                       例如 8 次采样中 3 次正确 → pass_rate = 0.375
        
        Returns:
            更新后的状态 dict: {p, lp, n_updates, weight}
            
        公式推导：
            p_new = beta * p_old + (1 - beta) * pass_rate
            lp = p_new - p_old  (正值 = 正在学会，负值 = 正在遗忘)
        """
        if sample_id not in self.states:
            # 首次遇到：用当前 pass_rate 初始化
            self.states[sample_id] = {
                'p': pass_rate,
                'lp': 0.0,
                'n_updates': 1,
            }
        else:
            state = self.states[sample_id]
            p_old = state['p']
            # EMA 更新：新值 = 历史权重 * 旧值 + 当前权重 * 新观测
            p_new = self.beta * p_old + (1.0 - self.beta) * pass_rate
            lp = p_new - p_old  # 学习进度
            state['p'] = p_new
            state['lp'] = lp
            state['n_updates'] += 1

        return self.states[sample_id]

    def compute_weight(self, sample_id: str) -> float:
        """
        计算某道题的 LP 权重（论文公式）

        公式：
          w_i(t) = sigmoid(κ * Δ_i(t)) + b

        其中：
          Δ_i(t) = lp = p_ema_new - p_ema_old（学习进度）
          κ = 8.0（陡峭度）
          b = 0.5（偏置）

        行为：
          Δ > 0（正在进步）→ weight > 1.0 → 放大梯度
          Δ = 0（无变化）  → weight = 1.0 → 中性
          Δ < 0（在退步）  → weight < 1.0 → 缩小梯度

        范围：[b, 1+b] = [0.5, 1.5]

        Args:
            sample_id: 题目唯一标识

        Returns:
            float: LP 权重，范围 [0.5, 1.5]
        """
        if sample_id not in self.states:
            return 1.0  # 未知题目返回中性权重

        state = self.states[sample_id]
        lp = state['lp']  # 学习进度 = p_new - p_old

        # 论文公式：w = sigmoid(κ * Δ) + b
        sigmoid_val = 1.0 / (1.0 + math.exp(-self.kappa * lp))
        weight = sigmoid_val + self.b

        return weight

    def batch_update_and_get_weights(
        self,
        sample_ids: np.ndarray,
        rewards: np.ndarray,
        rollout_n: int = 8,
        is_pg: np.ndarray = None,
    ) -> np.ndarray:
        """
        批量处理一个 training step 的所有样本

        这是 Hook B 的核心入口：
          1. 按 sample_id 分组，计算每题的 pass_rate
          2. 更新 EMA 状态（PG 样本不更新主 p_ema）
          3. 返回每个样本对应的 LP weight

        Args:
            sample_ids: shape (batch_size,)，每个样本的题目 ID
            rewards: shape (batch_size,)，每个样本的 reward (0 或 1)
            rollout_n: 每题的 rollout 次数
            is_pg: shape (batch_size,)，标记哪些是 PG 样本（可选）

        Returns:
            weights: shape (batch_size,)，每个样本的 LP weight

        PG 隔离原则：
            PG 样本的 pass_rate 不更新主 p_ema，因为有前缀脚手架不代表模型真实能力。
            PG 样本单独更新 pg_p_ema 和 pg_seen（用于监控 PG 效果）。
        """
        if is_pg is None:
            is_pg = np.zeros(len(sample_ids), dtype=bool)

        # Step 1: 按 (sample_id, is_pg) 分组，计算 pass_rate
        from collections import defaultdict
        normal_groups = defaultdict(list)  # sample_id → [rewards]
        pg_groups = defaultdict(list)      # sample_id → [rewards]

        for sid, r, pg in zip(sample_ids, rewards, is_pg):
            if pg:
                pg_groups[sid].append(float(r))
            else:
                normal_groups[sid].append(float(r))

        # Step 2a: Normal 样本更新主 p_ema
        for sid, reward_list in normal_groups.items():
            pass_rate = sum(1 for r in reward_list if r > 0) / len(reward_list)
            self.update(sid, pass_rate)

        # Step 2b: PG 样本更新独立的 pg_p_ema（不影响主 p_ema）
        for sid, reward_list in pg_groups.items():
            pass_rate = sum(1 for r in reward_list if r > 0) / len(reward_list)
            if sid not in self.states:
                self.states[sid] = {'p': 0.0, 'lp': 0.0, 'n_updates': 0}
            state = self.states[sid]
            # PG 独立统计
            pg_p_old = state.get('pg_p_ema', 0.0)
            state['pg_p_ema'] = self.beta * pg_p_old + (1.0 - self.beta) * pass_rate
            state['pg_seen'] = state.get('pg_seen', 0) + 1

        # Step 3: 构造 weight 数组（与 sample_ids 对齐，PG 和 normal 用相同 weight）
        weights = np.array([self.compute_weight(sid) for sid in sample_ids],
                          dtype=np.float32)
        return weights

    def get_state_summary(self) -> dict:
        """
        获取当前 LP 状态的统计摘要（用于 logging）
        
        Returns:
            dict with keys: n_problems, avg_p, avg_lp, 
                           hard_zero_count, sweet_spot_count, mastered_count
        """
        if not self.states:
            return {'n_problems': 0}

        ps = [s['p'] for s in self.states.values()]
        lps = [s['lp'] for s in self.states.values()]

        return {
            'n_problems': len(self.states),
            'avg_p': np.mean(ps),
            'avg_lp': np.mean(lps),
            'hard_zero_count': sum(1 for p in ps if p < 0.05),
            'sweet_spot_count': sum(1 for p in ps if 0.1 <= p <= 0.5),
            'mastered_count': sum(1 for p in ps if p > 0.8),
            'avg_weight': np.mean([self.compute_weight(sid)
                                   for sid in self.states.keys()]),
        }

    def get_problem_categories(self) -> dict:
        """
        将题目按 LP 状态分类（用于 Cycle 重采样）

        Returns:
            dict: {
                'hard_zero': [sample_ids...],     # p < 0.05，完全不会
                'struggling': [sample_ids...],    # 0.05 <= p < 0.15，正在挣扎
                'learning': [sample_ids...],      # lp > 0 且 0.15 <= p <= 0.6，正在学习
                'sweet_spot': [sample_ids...],    # 0.1 <= p <= 0.5，最佳难度区
                'mastered': [sample_ids...],      # p > 0.8，已掌握
            }
        """
        categories = {
            'hard_zero': [],
            'struggling': [],
            'learning': [],
            'sweet_spot': [],
            'mastered': [],
        }
        for sid, state in self.states.items():
            p = state['p']
            lp = state['lp']
            if p < 0.05:
                categories['hard_zero'].append(sid)
            elif p < 0.15:
                categories['struggling'].append(sid)
            elif p > 0.8:
                categories['mastered'].append(sid)
            elif 0.1 <= p <= 0.5:
                categories['sweet_spot'].append(sid)
                if lp > 0:
                    categories['learning'].append(sid)
            else:
                if lp > 0:
                    categories['learning'].append(sid)
        return categories

    def get_pg_candidates(self) -> list:
        """
        获取满足 PG (Prefix-Guided) 条件的 hard-zero 样本

        PG 触发条件（三个必须同时满足）：
          - p_ema <= 0.05：模型完全不会
          - n_updates >= 2：至少观测过 2 次（有初始 P0 算 1 次，训练中再见 1 次）
          - lp <= 0.01：没有明显进步趋势（排除"正在突破"的题）

        设计原理：
          - 如果只看 p_ema < 0.05 会误选"刚出现一次碰巧全错"的题
          - seen >= 2 确保这确实是"反复做不出"而非偶然
          - lp <= 0.01 排除正在好转的题（不需要 PG 辅助了）

        Returns:
            list of sample_id: 满足 PG 条件的题目 ID 列表
        """
        candidates = []
        for sid, state in self.states.items():
            p = state['p']
            lp = state['lp']
            seen = state.get('n_updates', 0)
            if p <= 0.05 and seen >= 2 and lp <= 0.01:
                candidates.append(sid)
        return candidates

    def save_state(self, path: str):
        """
        保存 LP 状态到 JSON 文件（Hook D 调用）
        
        保存内容包括：
          - 超参数配置（方便复现）
          - 所有题目的状态
          - 统计摘要
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'config': {
                'beta': self.beta,
                'kappa': self.kappa,
                'b': self.b,
            },
            'states': self.states,
            'summary': self.get_state_summary(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str):
        """
        从 JSON 文件恢复 LP 状态（训练恢复时调用）
        """
        if not os.path.exists(path):
            print(f"[LP] 状态文件不存在，跳过加载: {path}")
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        self.states = data['states']
        # 可选：恢复超参数（通常由配置文件控制，这里只恢复状态）
        print(f"[LP] 已恢复 {len(self.states)} 道题的 LP 状态")
        return True

    def apply_lp_weights_to_advantages(
        self,
        advantages,  # torch.Tensor, shape (batch_size, seq_len)
        sample_ids: np.ndarray,   # shape (batch_size,)
        rewards: np.ndarray,      # shape (batch_size,)，用于更新状态
        rollout_n: int = 8,
    ):
        """
        Hook B 的完整入口：更新状态 + 加权 advantage
        
        调用位置：ray_trainer.py 中 compute_advantage() 之后
        
        流程：
          1. 批量更新 LP 状态（根据本轮 rewards）
          2. 计算每个样本的 LP weight
          3. 将 weight 广播到 token 维度，乘以 advantage
          
        Args:
            advantages: (batch_size, seq_len) 的 advantage tensor
            sample_ids: (batch_size,) 的样本 ID
            rewards: (batch_size,) 的 reward 值
            rollout_n: 每题 rollout 次数
            
        Returns:
            weighted_advantages: 加权后的 advantage tensor（同 shape）
        """
        import torch

        # 计算 LP weights
        weights = self.batch_update_and_get_weights(sample_ids, rewards, rollout_n)

        # 转为 torch tensor 并广播到 seq_len 维度
        weight_tensor = torch.tensor(weights, dtype=advantages.dtype,
                                     device=advantages.device)
        # shape: (batch_size,) → (batch_size, 1) 用于广播乘法
        weight_tensor = weight_tensor.unsqueeze(-1)

        # 逐 token 加权
        weighted_advantages = advantages * weight_tensor

        return weighted_advantages


# ===================== 单元测试 =====================

def _test_basic():
    """基础功能测试（论文公式 w = sigmoid(κΔ) + b）"""
    mgr = LPStateManager(beta=0.8, kappa=8.0, b=0.5)

    # 模拟一道题被多次更新
    # 第1次：pass_rate = 0/8 = 0
    mgr.update("q001", 0.0)
    assert mgr.states["q001"]['p'] == 0.0

    # 第2次：pass_rate = 2/8 = 0.25
    mgr.update("q001", 0.25)
    # p_new = 0.8 * 0.0 + 0.2 * 0.25 = 0.05
    assert abs(mgr.states["q001"]['p'] - 0.05) < 1e-6
    # lp = 0.05 - 0.0 = 0.05（正向进步）
    assert abs(mgr.states["q001"]['lp'] - 0.05) < 1e-6

    # 权重应该 > 1.0（正在进步，lp > 0）
    # w = sigmoid(8 * 0.05) + 0.5 = sigmoid(0.4) + 0.5 ≈ 0.598 + 0.5 = 1.098
    w = mgr.compute_weight("q001")
    assert w > 1.0, f"正在学习的题权重应该 > 1.0，实际={w:.3f}"

    # 模拟一道停滞的题（lp = 0）
    mgr.states["q002"] = {'p': 0.50, 'lp': 0.0, 'n_updates': 10}
    w_neutral = mgr.compute_weight("q002")
    # w = sigmoid(8 * 0) + 0.5 = 0.5 + 0.5 = 1.0
    assert abs(w_neutral - 1.0) < 1e-6, f"lp=0 的题权重应该 = 1.0，实际={w_neutral:.3f}"

    # 模拟一道在退步的题（lp < 0）
    mgr.states["q003"] = {'p': 0.30, 'lp': -0.05, 'n_updates': 5}
    w_regress = mgr.compute_weight("q003")
    # w = sigmoid(8 * -0.05) + 0.5 = sigmoid(-0.4) + 0.5 ≈ 0.402 + 0.5 = 0.902
    assert w_regress < 1.0, f"退步的题权重应该 < 1.0，实际={w_regress:.3f}"

    print("✅ 基础测试通过（论文公式）")
    print(f"  正在进步 (lp=+0.05): weight = {w:.3f}")
    print(f"  停滞 (lp=0):         weight = {w_neutral:.3f}")
    print(f"  退步 (lp=-0.05):     weight = {w_regress:.3f}")


def _test_batch():
    """批量处理测试"""
    mgr = LPStateManager()

    # 模拟 batch: 2道题，每题 rollout_n=4
    sample_ids = np.array(["q001", "q001", "q001", "q001",
                           "q002", "q002", "q002", "q002"])
    rewards = np.array([0, 1, 0, 1,   # q001: pass_rate = 0.5
                        1, 1, 1, 0])   # q002: pass_rate = 0.75

    weights = mgr.batch_update_and_get_weights(sample_ids, rewards, rollout_n=4)
    assert weights.shape == (8,)
    # 同一题的所有 rollout 应有相同权重
    assert weights[0] == weights[1] == weights[2] == weights[3]
    assert weights[4] == weights[5] == weights[6] == weights[7]

    print("✅ 批量测试通过")
    print(f"  q001 (pass=0.5) weight: {weights[0]:.3f}")
    print(f"  q002 (pass=0.75) weight: {weights[4]:.3f}")


def _test_save_load():
    """序列化测试"""
    import tempfile

    mgr = LPStateManager()
    mgr.update("q001", 0.3)
    mgr.update("q002", 0.7)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name

    mgr.save_state(path)

    # 新建一个 manager 并恢复
    mgr2 = LPStateManager()
    mgr2.load_state(path)

    assert mgr2.states["q001"]['p'] == mgr.states["q001"]['p']
    assert mgr2.states["q002"]['p'] == mgr.states["q002"]['p']

    os.unlink(path)
    print("✅ 序列化测试通过")


if __name__ == "__main__":
    _test_basic()
    _test_batch()
    _test_save_load()
    print("\n所有测试通过 ✅")
