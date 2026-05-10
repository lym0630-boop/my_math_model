"""
LPPO (Learning Progress PPO) 模块

核心思路：根据每道题的学习进度 (Learning Progress) 动态调整 advantage 的权重，
让模型更多关注"正在学会"的题目，减少对已掌握或完全不会的题目的无效训练。

模块组成：
  - lp_state_manager: LP 核心状态管理（EMA pass rate, LP weight 计算）
  - lp_init: 从历史 student_responses 初始化 P0
  - prefix_guided_rollout: 为 hard-zero 题准备带前缀的 prompt（复用 warmstart 逻辑）
  - build_cycle_data: 基于 LP state 的 Cycle 重采样
"""
