"""
Cycle 重采样模块 — 基于 LP State 动态调整训练数据组成

功能：
  在每个 training cycle（或固定 step 间隔）结束后，根据 LP state
  重新构造下一 cycle 的训练数据，实现：
    1. 过采样 "正在学习" 的题目（learning，lp > 0）
    2. 欠采样 "已掌握" 的题目（mastered，p > 0.8）
    3. 为 "hard-zero" 的题目注入 PG 样本
    4. 保持一定比例的新题（exploration）

设计原理：
  标准 GRPO 对所有题一视同仁地采样，这导致：
    - 已掌握的简单题：reward 恒为 1，advantage ≈ 0，浪费 GPU 算力
    - 完全不会的难题：reward 恒为 0，advantage ≈ 0，无学习信号
    - 正在学习的题：有正负 reward 对比，advantage 有意义
  
  Cycle 重采样让训练资源集中在最有价值的题目上。

与 prepare_math_rlvr_data.py 的关系：
  - prepare_math_rlvr_data.py 做一次性的静态数据准备
  - build_cycle_data.py 在训练过程中动态调整数据组成

使用场景：
  在 run_math_grpo.sh 中，每个 cycle 结束后调用：
    python3 -m lppo.build_cycle_data --lp_state_path ... --output_dir ...
"""

import json
import os
import random
import argparse
import hashlib
from collections import Counter

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_state_manager import LPStateManager
from lp_init import compute_sample_id


# 默认数据配比
DEFAULT_RATIOS = {
    'learning': 0.40,     # 正在学习的题（核心训练目标）
    'sweet_spot': 0.25,   # 甜区题（保持稳定训练信号）
    'struggling': 0.15,   # 挣扎中的题（给它们一些机会）
    'hard_zero_pg': 0.10, # hard-zero 的 PG 样本
    'mastered': 0.05,     # 已掌握的题（防遗忘）
    'exploration': 0.05,  # 未见过的新题
}

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_cycle_data(
    lp_state_path: str,
    original_data_path: str,
    output_path: str,
    reference_answers_path: str = None,
    target_size: int = None,
    ratios: dict = None,
    seed: int = 42,
):
    """
    基于 LP state 构建下一 cycle 的训练数据
    
    Args:
        lp_state_path: LP state JSON 文件路径
        original_data_path: 原始训练数据 parquet 路径
        output_path: 输出的新 cycle 数据 parquet 路径
        reference_answers_path: 参考解文件路径（PG 样本需要）
        target_size: 目标数据量（默认与原始相同）
        ratios: 各类别数据配比（默认使用 DEFAULT_RATIOS）
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    if ratios is None:
        ratios = DEFAULT_RATIOS

    # 加载 LP state
    lp_mgr = LPStateManager()
    if not lp_mgr.load_state(lp_state_path):
        print("[Cycle] LP state 不存在，使用原始数据")
        return

    # 加载原始训练数据
    original_df = pd.read_parquet(original_data_path)
    if target_size is None:
        target_size = len(original_df)
    print(f"[Cycle] 原始数据: {len(original_df)} 行, 目标: {target_size} 行")

    # 获取题目分类
    categories = lp_mgr.get_problem_categories()
    print(f"[Cycle] LP 分类:")
    for cat, ids in categories.items():
        print(f"  {cat}: {len(ids)}")

    # 构建 sample_id → original record 的映射
    id_to_records = {}
    for idx, row in original_df.iterrows():
        extra = row.get('extra_info', {})
        if isinstance(extra, str):
            extra = json.loads(extra)
        sid = extra.get('sample_id')
        if sid:
            id_to_records[sid] = row.to_dict()

    # 按配比采样
    cycle_records = []

    for cat, ratio in ratios.items():
        n_target = int(target_size * ratio)

        if cat == 'hard_zero_pg':
            # PG 样本需要特殊处理（由 prefix_guided_rollout 生成）
            # 这里先跳过，后面单独处理
            continue
        elif cat == 'exploration':
            # 从不在 LP state 中的题目采样
            known_ids = set(lp_mgr.states.keys())
            unknown_records = [
                row.to_dict() for _, row in original_df.iterrows()
                if _get_sample_id(row) not in known_ids
            ]
            random.shuffle(unknown_records)
            cycle_records.extend(unknown_records[:n_target])
        else:
            # 从对应类别中采样
            cat_ids = categories.get(cat, [])
            random.shuffle(cat_ids)
            sampled = 0
            for sid in cat_ids:
                if sampled >= n_target:
                    break
                if sid in id_to_records:
                    cycle_records.append(id_to_records[sid])
                    sampled += 1

    # 处理 PG 样本（使用精确的 PG 候选条件：p<=0.05, seen>=2, lp<=0.01）
    pg_target = int(target_size * ratios.get('hard_zero_pg', 0.10))
    if pg_target > 0 and reference_answers_path:
        pg_candidate_ids = lp_mgr.get_pg_candidates()
        pg_records = _generate_pg_samples(
            pg_candidate_ids,
            id_to_records,
            reference_answers_path,
            pg_target,
        )
        cycle_records.extend(pg_records)
        print(f"[Cycle] 生成 PG 样本: {len(pg_records)}")

    # 如果数量不足，从 sweet_spot 补充
    if len(cycle_records) < target_size:
        shortfall = target_size - len(cycle_records)
        sweet_ids = categories.get('sweet_spot', []) + categories.get('learning', [])
        random.shuffle(sweet_ids)
        for sid in sweet_ids:
            if len(cycle_records) >= target_size:
                break
            if sid in id_to_records:
                cycle_records.append(id_to_records[sid])

    # 打乱并保存
    random.shuffle(cycle_records)
    cycle_df = pd.DataFrame(cycle_records[:target_size])
    cycle_df.to_parquet(output_path)

    print(f"[Cycle] 已保存 {len(cycle_df)} 行到: {output_path}")

    # 统计
    _print_cycle_stats(cycle_records[:target_size], lp_mgr)


def _get_sample_id(row) -> str:
    """从 DataFrame row 中提取 sample_id"""
    extra = row.get('extra_info', {})
    if isinstance(extra, str):
        extra = json.loads(extra)
    if isinstance(extra, dict):
        return extra.get('sample_id', '')
    return ''


def _generate_pg_samples(
    hard_zero_ids: list,
    id_to_records: dict,
    reference_answers_path: str,
    max_samples: int,
) -> list:
    """为 hard-zero 题目生成 PG 样本"""
    from prefix_guided_rollout import prepare_pg_prompt

    # 加载参考解
    ref_answers = {}
    if os.path.exists(reference_answers_path):
        with open(reference_answers_path) as f:
            for line in f:
                d = json.loads(line)
                ref_answers[d['question'][:120]] = d.get('answer', '')

    pg_records = []
    for sid in hard_zero_ids:
        if len(pg_records) >= max_samples:
            break
        if sid not in id_to_records:
            continue

        record = id_to_records[sid]
        question = _extract_question(record)
        gt = _extract_ground_truth(record)
        qk = question[:120]

        if qk not in ref_answers:
            continue

        result = prepare_pg_prompt(question, ref_answers[qk], gt)
        if result is None:
            continue

        pg_record = dict(record)  # 复制原始记录
        pg_record['prompt'] = result['prompt']
        extra = pg_record.get('extra_info', {})
        if isinstance(extra, str):
            extra = json.loads(extra)
        extra['is_pg_sample'] = True
        extra['prefix_ratio'] = result['prefix_ratio']
        pg_record['extra_info'] = extra
        pg_records.append(pg_record)

    return pg_records


def _extract_question(record: dict) -> str:
    """从记录中提取问题文本"""
    prompt = record.get('prompt', [])
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')
    return ''


def _extract_ground_truth(record: dict) -> str:
    """从记录中提取标准答案"""
    rm = record.get('reward_model', {})
    if isinstance(rm, str):
        rm = json.loads(rm)
    return rm.get('ground_truth', '')


def _print_cycle_stats(records: list, lp_mgr: LPStateManager):
    """打印 cycle 数据统计"""
    pg_count = sum(1 for r in records
                   if isinstance(r.get('extra_info'), dict)
                   and r['extra_info'].get('is_pg_sample'))
    total = len(records)
    print(f"\n[Cycle] 数据统计:")
    print(f"  总量: {total}")
    print(f"  PG 样本: {pg_count} ({pg_count/max(total,1)*100:.1f}%)")
    print(f"  普通样本: {total - pg_count}")


def main():
    parser = argparse.ArgumentParser(description='基于 LP State 构建 Cycle 训练数据')
    parser.add_argument('--lp_state_path', required=True,
                        help='LP state JSON 文件路径')
    parser.add_argument('--original_data', required=True,
                        help='原始训练 parquet 路径')
    parser.add_argument('--output', required=True,
                        help='输出 parquet 路径')
    parser.add_argument('--reference_answers',
                        default=None,
                        help='参考解 jsonl 文件（PG 样本用）')
    parser.add_argument('--target_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    build_cycle_data(
        lp_state_path=args.lp_state_path,
        original_data_path=args.original_data,
        output_path=args.output,
        reference_answers_path=args.reference_answers,
        target_size=args.target_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
