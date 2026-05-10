"""
构建 Active Pool — 从 120K 题池中按 P0 分桶选出固定训练子集

功能：
  根据 student_responses 中的历史 pass_rate (P0) 将题目分桶，
  按照 LPPO 训练需要的难度分布选出固定的 active pool。

设计原理：
  LPPO 需要每道题被多次观测才能计算有效的 learning progress，
  所以不能用 120K 全量（每题只见 1 次），而是用小题池多 epoch。

  题池大小选择 3000 题：
    - batch_size=8 → 每 epoch 375 步
    - 3 epoch 每题见 3 次 → LP 有 3 次 EMA 更新
    - 总训练 ~1125 步 ≈ 16 小时

使用方式：
  python3 -m lppo.build_active_pool \
    --questions sft_data/dpo_questions_120k_expanded_v4.jsonl \
    --student_responses sft_data/student_responses_120k_expanded_v4.jsonl \
    --teacher_responses sft_data/teacher_responses_48k_final_v4.jsonl \
    --output_dir data/lppo_active_pool \
    --pool_size 3000 \
    --pg_ratio 0.20
"""

import json
import os
import re
import random
import hashlib
import argparse
import logging
from collections import Counter, defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE_DIR = '/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline'
SFT_DIR = os.path.join(BASE_DIR, 'sft_data')

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def make_sample_id(question: str, ground_truth: str) -> str:
    """与 prepare_math_rlvr_data.py 和 lp_init.py 一致的 hash"""
    key = question.strip() + "||" + ground_truth.strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def build_active_pool(
    questions_path: str,
    student_responses_path: str,
    teacher_responses_path: str,
    output_dir: str,
    pool_size: int = 3000,
    pg_ratio: float = 0.20,
    seed: int = 42,
):
    """
    从题池中按 P0 分桶选出 active pool，生成 train.parquet

    分桶策略（基于 P0 = student_responses 的 pass_rate）：
      sweet_mid (P0 0.10-0.60):  ~60%  最有价值的训练区间
      hard (P0 ≤ 0.05):         ~17%  hard-zero，部分转 PG
      easy (P0 ≥ 0.80):         ~7%   防遗忘
      exploration (其余):        ~16%  补充覆盖

    PG 样本从 hard 桶中选取有 reference answer 的题目生成。
    """
    random.seed(seed)

    # 1. 加载 student_responses → 每题的 P0
    logging.info("加载 student_responses...")
    student_p0 = {}  # question_key → pass_rate
    student_data = {}  # question_key → full record
    with open(student_responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qk = d['question'][:120]
            p0 = d['num_correct'] / max(d['num_total'], 1)
            student_p0[qk] = p0
            student_data[qk] = d
    logging.info(f"  已加载 {len(student_p0)} 题的 P0")

    # 2. 加载 questions（获取 ground_truth 和 answer）
    logging.info("加载题目文件...")
    questions = {}  # question_key → {question, ground_truth, answer}
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
    logging.info(f"  已加载 {len(questions)} 题")

    # 3. 加载 teacher_responses（判断 hard 题是否有 reference）
    logging.info("加载 teacher_responses...")
    teacher_status = {}
    if os.path.exists(teacher_responses_path):
        with open(teacher_responses_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                qk = d['question'][:120]
                teacher_status[qk] = d.get('status', 'unknown')
    logging.info(f"  已加载 {len(teacher_status)} 题的 teacher 状态")

    # 4. 按 P0 分桶
    buckets = {
        'sweet_mid': [],   # P0 ∈ [0.10, 0.60]
        'hard': [],        # P0 ≤ 0.05
        'easy': [],        # P0 ≥ 0.80
        'exploration': [], # 其余 (0.05, 0.10) ∪ (0.60, 0.80) 或无 P0
    }

    for qk, q_data in questions.items():
        p0 = student_p0.get(qk)
        if p0 is None:
            buckets['exploration'].append(qk)
        elif p0 <= 0.05:
            buckets['hard'].append(qk)
        elif 0.10 <= p0 <= 0.60:
            buckets['sweet_mid'].append(qk)
        elif p0 >= 0.80:
            buckets['easy'].append(qk)
        else:
            buckets['exploration'].append(qk)

    logging.info("P0 分桶统计:")
    for bucket, items in buckets.items():
        logging.info(f"  {bucket}: {len(items)}")

    # 5. 按配比从每个桶采样
    target_counts = {
        'sweet_mid': int(pool_size * 0.60),    # 1800
        'hard': int(pool_size * 0.17),          # 510
        'easy': int(pool_size * 0.07),          # 210
        'exploration': int(pool_size * 0.16),   # 480
    }

    selected_keys = []
    for bucket, target in target_counts.items():
        pool = buckets[bucket]
        random.shuffle(pool)
        # hard 桶优先选有 teacher reference 的题（方便后续做 PG）
        if bucket == 'hard':
            has_ref = [qk for qk in pool if teacher_status.get(qk) in ('both_correct', 'one_correct')]
            no_ref = [qk for qk in pool if qk not in set(has_ref)]
            pool = has_ref + no_ref
        selected = pool[:target]
        selected_keys.extend(selected)
        logging.info(f"  {bucket}: 选取 {len(selected)} / {len(pool)}")

    # 补足到 pool_size
    shortfall = pool_size - len(selected_keys)
    if shortfall > 0:
        remaining = [qk for bucket in buckets.values() for qk in bucket
                     if qk not in set(selected_keys)]
        random.shuffle(remaining)
        selected_keys.extend(remaining[:shortfall])

    selected_keys = selected_keys[:pool_size]
    random.shuffle(selected_keys)
    logging.info(f"Active Pool 最终大小: {len(selected_keys)}")

    # 6. 确定 PG 样本
    # 从 hard 桶中有 reference answer 的题目生成 PG prompt
    pg_count = int(pool_size * pg_ratio)
    hard_with_ref = [qk for qk in selected_keys
                     if student_p0.get(qk, 1.0) <= 0.05
                     and questions[qk]['answer']
                     and len(questions[qk]['answer']) > 100]

    random.shuffle(hard_with_ref)
    pg_keys = set(hard_with_ref[:pg_count])
    logging.info(f"PG 样本: {len(pg_keys)} / {pg_count} 目标")

    # 7. 构造 parquet 记录
    from lppo.prefix_guided_rollout import prepare_pg_prompt

    records = []
    pg_success = 0
    pg_fail = 0

    for qk in selected_keys:
        q_data = questions[qk]
        question = q_data['question']
        gt = q_data['ground_truth']
        sample_id = make_sample_id(question, gt)

        if qk in pg_keys:
            # 生成 PG prompt
            ref_answer = q_data['answer']
            result = prepare_pg_prompt(question, ref_answer, gt)
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
                        "student_p0": student_p0.get(qk),
                    },
                })
                pg_success += 1
                continue
            else:
                pg_fail += 1
                # fallback: 作为 normal 样本

        # Normal 样本
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
                "student_p0": student_p0.get(qk),
            },
        })

    logging.info(f"PG 生成: 成功 {pg_success}, 失败 {pg_fail}")

    # 8. 保存
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(records)

    # 分 train/test (保留 200 题做 val)
    num_test = min(200, len(records) // 10)
    train_records = records[:len(records) - num_test]
    test_records = records[len(records) - num_test:]

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    logging.info(f"保存: {train_path} ({len(train_df)} 行)")
    logging.info(f"保存: {test_path} ({len(test_df)} 行)")

    # 9. 保存 active pool 元信息（用于后续 refresh）
    meta = {
        'pool_size': len(selected_keys),
        'pg_ratio': pg_ratio,
        'pg_count': pg_success,
        'selected_keys': selected_keys,
        'bucket_stats': {k: len(v) for k, v in buckets.items()},
        'target_counts': target_counts,
    }
    meta_path = os.path.join(output_dir, "pool_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info(f"元信息: {meta_path}")

    # 统计
    pg_in_train = sum(1 for r in train_records
                      if r['extra_info'].get('is_pg_sample'))
    logging.info(f"\n=== Active Pool 统计 ===")
    logging.info(f"  总量: {len(records)}")
    logging.info(f"  训练集: {len(train_records)} (PG: {pg_in_train})")
    logging.info(f"  测试集: {len(test_records)}")
    logging.info(f"  PG 实际占比: {pg_in_train/len(train_records)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='构建 LPPO Active Pool')
    parser.add_argument('--questions',
                        default=os.path.join(SFT_DIR, 'dpo_questions_120k_expanded_v4.jsonl'))
    parser.add_argument('--student_responses',
                        default=os.path.join(SFT_DIR, 'student_responses_120k_expanded_v4.jsonl'))
    parser.add_argument('--teacher_responses',
                        default=os.path.join(SFT_DIR, 'teacher_responses_48k_final_v4.jsonl'))
    parser.add_argument('--output_dir',
                        default=os.path.join(BASE_DIR, 'data/lppo_active_pool'))
    parser.add_argument('--pool_size', type=int, default=3000)
    parser.add_argument('--pg_ratio', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    build_active_pool(
        questions_path=args.questions,
        student_responses_path=args.student_responses,
        teacher_responses_path=args.teacher_responses,
        output_dir=args.output_dir,
        pool_size=args.pool_size,
        pg_ratio=args.pg_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
