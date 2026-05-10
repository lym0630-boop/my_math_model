"""
PG Refresh — 每 round 之间根据最新 LP state 刷新 PG 样本

功能：
  读取训练中更新后的 lp_state.json，重新判断哪些题是 hard-zero，
  为它们生成新的 PG prompt，替换 parquet 中的 PG 样本。

  核心逻辑：
    - 从 pool_meta.json 获取 active pool 的固定题目列表
    - 从 lp_state.json 获取最新 p_ema
    - 满足 PG 条件 (p_ema <= 0.05, seen >= 2, lp <= 0.01) 的题 → 生成 PG
    - 已经学会的题 (p_ema > 0.05) → 退出 PG，变回 normal
    - 输出新的 train.parquet

使用方式：
  python3 -m lppo.refresh_pg \
    --pool_dir data/lppo_active_pool \
    --lp_state_path checkpoints/exp/lp_state.json \
    --pg_ratio 0.15 \
    --output_dir data/lppo_active_pool_round2
"""

import json
import os
import random
import hashlib
import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE_DIR = '/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline'
SFT_DIR = os.path.join(BASE_DIR, 'sft_data')
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def make_sample_id(question: str, ground_truth: str) -> str:
    key = question.strip() + "||" + ground_truth.strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def refresh_pg(
    pool_dir: str,
    lp_state_path: str,
    questions_path: str,
    pg_ratio: float = 0.15,
    output_dir: str = None,
    seed: int = 42,
):
    """
    根据最新 LP state 刷新 PG 样本

    流程：
      1. 加载 pool_meta.json 获取固定题目列表
      2. 加载 lp_state.json 获取最新 p_ema
      3. 用 get_pg_candidates() 条件判断新的 PG 集合
      4. 生成新 PG prompt + normal prompt → 新 parquet
    """
    random.seed(seed)
    if output_dir is None:
        output_dir = pool_dir

    # 1. 加载 pool 元信息
    meta_path = os.path.join(pool_dir, "pool_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    selected_keys = meta['selected_keys']
    pool_size = len(selected_keys)
    logging.info(f"Active Pool: {pool_size} 题")

    # 2. 加载 LP state
    from lppo.lp_state_manager import LPStateManager
    lp_mgr = LPStateManager()
    lp_mgr.load_state(lp_state_path)
    logging.info(f"LP State: {len(lp_mgr.states)} 条记录")

    # 3. 加载题目数据
    questions = {}
    with open(questions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qk = d['question'][:120]
            questions[qk] = {
                'question': d['question'],
                'ground_truth': d.get('ground_truth', ''),
                'answer': d.get('answer', ''),
            }

    # 4. 判断哪些题应该是 PG
    # 用 LP state 的精确条件：p_ema <= 0.05, seen >= 2, lp <= 0.01
    pg_target = int(pool_size * pg_ratio)
    pg_candidates = []

    for qk in selected_keys:
        q_data = questions.get(qk)
        if not q_data:
            continue
        sample_id = make_sample_id(q_data['question'], q_data['ground_truth'])
        state = lp_mgr.states.get(sample_id)
        if state is None:
            continue
        p = state.get('p', 1.0)
        lp = state.get('lp', 0.0)
        seen = state.get('n_updates', 0)
        # PG 条件
        if p <= 0.05 and seen >= 2 and lp <= 0.01:
            # 还需要有 reference answer
            if q_data['answer'] and len(q_data['answer']) > 100:
                pg_candidates.append(qk)

    random.shuffle(pg_candidates)
    pg_keys = set(pg_candidates[:pg_target])
    logging.info(f"PG 候选: {len(pg_candidates)}, 选取: {len(pg_keys)} (目标 {pg_target})")

    # 5. 统计变化
    # 加载旧 parquet 看之前有哪些是 PG
    old_train_path = os.path.join(pool_dir, "train.parquet")
    if os.path.exists(old_train_path):
        old_df = pd.read_parquet(old_train_path)
        old_pg_count = sum(1 for _, row in old_df.iterrows()
                          if isinstance(row.get('extra_info'), dict)
                          and row['extra_info'].get('is_pg_sample', False))
        logging.info(f"旧 PG 数量: {old_pg_count} → 新 PG 数量: {len(pg_keys)}")

    # 6. 生成新 parquet
    from lppo.prefix_guided_rollout import prepare_pg_prompt

    records = []
    pg_success = 0

    for qk in selected_keys:
        q_data = questions.get(qk)
        if not q_data:
            continue
        question = q_data['question']
        gt = q_data['ground_truth']
        sample_id = make_sample_id(question, gt)

        if qk in pg_keys:
            result = prepare_pg_prompt(question, q_data['answer'], gt)
            if result is not None:
                records.append({
                    "data_source": "math",
                    "prompt": result["prompt"],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": gt},
                    "extra_info": {
                        "sample_id": sample_id,
                        "is_pg_sample": True,
                        "prefix_ratio": result["prefix_ratio"],
                        "prefix_level": result["prefix_level"],
                        "prefix_len": result["prefix_len"],
                    },
                })
                pg_success += 1
                continue

        # Normal
        records.append({
            "data_source": "math",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": ""},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "sample_id": sample_id,
                "is_pg_sample": False,
            },
        })

    # 7. 保存
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(records)

    num_test = min(200, len(records) // 10)
    train_records = records[:len(records) - num_test]
    test_records = records[len(records) - num_test:]

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    pg_in_train = sum(1 for r in train_records if r['extra_info'].get('is_pg_sample'))
    logging.info(f"\n=== Refresh 结果 ===")
    logging.info(f"  训练集: {len(train_records)} (PG: {pg_in_train}, {pg_in_train/len(train_records)*100:.1f}%)")
    logging.info(f"  测试集: {len(test_records)}")
    logging.info(f"  保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='刷新 PG 样本')
    parser.add_argument('--pool_dir', required=True, help='Active Pool 目录')
    parser.add_argument('--lp_state_path', required=True, help='最新 LP state 路径')
    parser.add_argument('--questions',
                        default=os.path.join(SFT_DIR, 'dpo_questions_120k_expanded_v4.jsonl'))
    parser.add_argument('--pg_ratio', type=float, default=0.15)
    parser.add_argument('--output_dir', default=None,
                        help='输出目录（默认覆盖 pool_dir）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    refresh_pg(
        pool_dir=args.pool_dir,
        lp_state_path=args.lp_state_path,
        questions_path=args.questions,
        pg_ratio=args.pg_ratio,
        output_dir=args.output_dir or args.pool_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
