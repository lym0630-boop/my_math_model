"""
准备数学 RLVR 训练数据（verl parquet 格式）

从 extracted_qa.jsonl 筛选题目，生成 verl 需要的 parquet 文件。

数据格式要求（每行）：
  - data_source: str  → 路由到 reward function
  - prompt: list[dict] → 对话格式 [{"role":"user","content":...}, {"role":"assistant","content":""}]
  - ability: str       → 标签
  - reward_model: dict → {"style":"rule", "ground_truth": str}
  - extra_info: dict   → 可选

Usage:
  python3 prepare_math_rlvr_data.py --output_dir data/parquet_math_rlvr
"""

import argparse
import json
import os
import re
import random
import hashlib
import logging
from collections import Counter, defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE_DIR = '/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline'
SFT_DIR = os.path.join(BASE_DIR, 'sft_data')

# 参考 select_dpo_questions.py 的学科分类
MATH_CATEGORIES = {
    '代数/函数': ['equation', 'solve for', 'roots', 'polynomial', 'quadratic',
                 'factori', 'inequality', 'function', 'logarithm', 'exponential'],
    '几何': ['triangle', 'circle', 'angle', 'polygon', 'area of', 'perimeter',
             'geometry', 'coordinate', 'volume', 'surface area'],
    '概率统计': ['probability', 'random variable', 'distribution', 'expectation',
                'variance', 'binomial', 'normal distribution'],
    '数列级数': ['sequence', 'series', 'converge', 'sum of', 'fibonacci', 'recurrence'],
    '微积分': ['integral', 'derivative', 'limit', 'calculus', 'differentiat'],
    '组合数学': ['combinat', 'permutation', 'choose', 'how many ways', 'counting'],
    '数论': ['number theory', 'prime', 'divisib', 'modular arithmetic', 'gcd', 'remainder'],
}

EXCLUDE_CATEGORIES = {
    '物理': ['force', 'velocity', 'acceleration', 'newton', 'energy', 'momentum',
             'circuit', 'voltage', 'kinetic', 'magnetic', 'electric field'],
    'CS/算法': ['algorithm', 'complexity', 'big-o', 'sorting', 'binary search',
               'dynamic programming', 'data structure'],
}

CATEGORY_ORDER = list(MATH_CATEGORIES.keys()) + ['其他数学']
SYSTEM_PROMPT_COT = "Please reason step by step, and put your final answer within \\boxed{}."


def make_sample_id(question: str, ground_truth: str) -> str:
    """
    基于题目+答案生成唯一且稳定的 sample_id

    设计考虑：
      - 跨 cycle 稳定：同一题无论重采样多少次，ID 不变
      - 加入 ground_truth：防止同一题目但不同 GT 版本产生冲突
      - SHA1 前 16 位：64-bit 空间，对 120K 题库碰撞概率 < 1e-8

    与 lppo/lp_init.py 中的 compute_sample_id() 完全一致。
    """
    key = question.strip() + "||" + ground_truth.strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def classify_question(question):
    q = question[:500].lower()
    for cat, keywords in EXCLUDE_CATEGORIES.items():
        if any(w in q for w in keywords):
            return cat
    for cat, keywords in MATH_CATEGORIES.items():
        if any(w in q for w in keywords):
            return cat
    return '其他数学'


def is_valid_gt(gt_str):
    gt = gt_str.strip()
    if re.match(r'^-?\d+$', gt): return True
    if re.match(r'^-?\d+\.\d{1,6}$', gt): return True
    if re.match(r'^-?\d+/\d+$', gt): return True
    return False


def difficulty_bucket(acc, teacher_status):
    if acc is None:
        return 'unknown'
    if 0.1 <= acc <= 0.5:
        return 'sweet_spot'
    if acc == 0 and teacher_status in {'both_correct', 'one_correct'}:
        return 'recoverable_hard'
    if 0.5 < acc < 0.8:
        return 'medium_easy'
    if 0.0 < acc < 0.1:
        return 'low_acc'
    return 'fallback'


def add_item(selected, selected_keys, selected_bucket_stats, item):
    qk = item['question'][:120].strip()
    if qk in selected_keys:
        return False
    selected.append(item)
    selected_keys.add(qk)
    selected_bucket_stats[difficulty_bucket(
        item.get('student_accuracy'), item.get('teacher_status'))] += 1
    return True


def select_balanced_by_category(candidates, fill_order, bucket_pools):
    """按类别尽量均衡采样：小类全保留，大类封顶到平均类别配额。"""
    category_pools = defaultdict(list)
    for item in candidates:
        category_pools[item['category']].append(item)

    categories = [cat for cat in CATEGORY_ORDER if category_pools.get(cat)]
    if not categories:
        return [], Counter()

    quota = max(1, len(candidates) // len(categories))
    logging.info("启用类别均衡采样: %d 个类别, 每类上限约 %d", len(categories), quota)

    selected = []
    selected_keys = set()
    selected_bucket_stats = Counter()
    selected_category_stats = Counter()

    for cat in categories:
        cat_items = category_pools[cat]
        cat_target = min(len(cat_items), quota)
        cat_selected = 0

        for bucket in fill_order:
            bucket_items = [
                item for item in bucket_pools.get(bucket, [])
                if item['category'] == cat
            ]
            for item in bucket_items:
                if cat_selected >= cat_target:
                    break
                if add_item(selected, selected_keys, selected_bucket_stats, item):
                    selected_category_stats[cat] += 1
                    cat_selected += 1
            if cat_selected >= cat_target:
                break

    logging.info("类别均衡后最终选取: %d / 原候选 %d", len(selected), len(candidates))
    logging.info("类别分布:")
    for cat, cnt in selected_category_stats.most_common():
        logging.info("  %s: %d", cat, cnt)

    return selected, selected_bucket_stats


def main():
    parser = argparse.ArgumentParser(description='准备数学 RLVR 训练数据')
    parser.add_argument('--extracted_qa', default=os.path.join(SFT_DIR, 'extracted_qa.jsonl'))
    parser.add_argument('--student_responses', default=os.path.join(SFT_DIR, 'student_responses_24k_v2.jsonl'),
                        help='用于过滤已知难度的题（可选）')
    parser.add_argument('--teacher_responses', default=os.path.join(SFT_DIR, 'teacher_responses_24k_v2.jsonl'),
                        help='用于识别 teacher 可救的 0% hard case（可选）')
    parser.add_argument('--output_dir', default=os.path.join(BASE_DIR, 'data/parquet_math_rlvr'))
    parser.add_argument('--total', type=int, default=15000,
                        help='总题数；<=0 表示使用所有通过过滤的候选')
    parser.add_argument('--balance_categories', action='store_true',
                        help='按数学题目类型尽量均衡采样；会对大类下采样')
    parser.add_argument('--num_test', type=int, default=1000, help='测试集数量')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. 加载 Student 采样结果（如果有），获取每题难度
    student_accuracy = {}
    if os.path.exists(args.student_responses):
        logging.info("加载 Student 采样结果用于难度估计...")
        with open(args.student_responses) as f:
            for line in f:
                d = json.loads(line)
                qk = d['question'][:120]
                student_accuracy[qk] = d['num_correct'] / d['num_total']
        logging.info("  已知难度的题: %d", len(student_accuracy))

    teacher_status = {}
    if os.path.exists(args.teacher_responses):
        logging.info("加载 Teacher 结果用于 recoverable hard case 识别...")
        with open(args.teacher_responses) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                teacher_status[d['question'][:120]] = d.get('status', 'unknown')
        logging.info("  已知 Teacher 状态的题: %d", len(teacher_status))

    # 2. 加载并筛选题目
    logging.info("加载题目文件: %s", args.extracted_qa)
    candidates = []
    seen_keys = set()
    filter_stats = Counter()

    with open(args.extracted_qa) as f:
        for line in f:
            d = json.loads(line)
            filter_stats['总数'] += 1

            # GT 来源。完整 extracted_qa 需要 code_output；精简题池可能没有 gt_source。
            gt_source = d.get('gt_source')
            if gt_source is not None and gt_source != 'code_output':
                filter_stats['排除: GT来源非code_output'] += 1
                continue

            gt = d.get('ground_truth')
            if not gt or not is_valid_gt(gt):
                filter_stats['排除: GT不可验证'] += 1
                continue

            # 题目长度
            qlen = len(d['question'])
            if qlen < 50 or qlen > 1500:
                filter_stats['排除: 题目长度'] += 1
                continue

            # 答案长度（不限上限，RLVR 不怕难题）
            alen = len(d['answer'])
            if alen < 100:
                filter_stats['排除: 答案太短'] += 1
                continue

            # 去重
            qk = d['question'][:120].strip()
            if qk in seen_keys:
                filter_stats['排除: 重复'] += 1
                continue
            seen_keys.add(qk)

            # 排除物理/CS
            cat = classify_question(d['question'])
            if cat in EXCLUDE_CATEGORIES:
                filter_stats[f'排除: {cat}'] += 1
                continue

            # 如果有 Student 难度信息，排除全对的题（advantage 全 0 没学习信号）
            acc = student_accuracy.get(qk)
            if acc is not None and acc >= 1.0:
                filter_stats['排除: Student全对'] += 1
                continue

            ts = teacher_status.get(qk, 'unknown')
            filter_stats['通过'] += 1
            candidates.append({
                'question': d['question'],
                'ground_truth': gt,
                'category': cat,
                'answer_length': alen,
                'student_accuracy': acc,  # 可能为 None
                'teacher_status': ts,
            })

    logging.info("筛选统计:")
    for k, v in filter_stats.most_common():
        logging.info("  %s: %d", k, v)

    # 3. 按显式难度配额采样
    bucket_ratios = {
        'sweet_spot': 0.55,       # 10%-50%，GRPO 最佳甜区
        'recoverable_hard': 0.20, # 0% 但 Teacher 可救
        'medium_easy': 0.15,      # 50%-80%，稳定锚点
        'low_acc': 0.05,          # 1%-9%，极难但非纯全错
        'unknown': 0.05,          # 未知难度，防止题池过窄
    }
    fill_order = ['sweet_spot', 'recoverable_hard', 'medium_easy', 'low_acc', 'unknown', 'fallback']
    bucket_pools = defaultdict(list)
    for item in candidates:
        bucket = difficulty_bucket(item.get('student_accuracy'), item.get('teacher_status'))
        bucket_pools[bucket].append(item)

    for bucket, pool in bucket_pools.items():
        random.shuffle(pool)
        if bucket == 'recoverable_hard':
            teacher_priority = {'both_correct': 0, 'one_correct': 1}
            pool.sort(key=lambda x: teacher_priority.get(x.get('teacher_status'), 9))

    selected = []
    selected_keys = set()
    selected_bucket_stats = Counter()
    if args.balance_categories:
        selected, selected_bucket_stats = select_balanced_by_category(
            candidates, fill_order, bucket_pools)
    elif args.total <= 0:
        logging.info("使用全量候选，不按 total 做下采样")
        for bucket in fill_order:
            for item in bucket_pools.get(bucket, []):
                add_item(selected, selected_keys, selected_bucket_stats, item)
    else:
        for bucket, ratio in bucket_ratios.items():
            target_n = int(args.total * ratio)
            pool = bucket_pools.get(bucket, [])
            taken = 0
            for item in pool:
                if len(selected) >= args.total or taken >= target_n:
                    break
                if add_item(selected, selected_keys, selected_bucket_stats, item):
                    taken += 1

        if len(selected) < args.total:
            shortfall = args.total - len(selected)
            logging.info("主桶不足，回填 %d 题...", shortfall)
            for bucket in fill_order:
                for item in bucket_pools.get(bucket, []):
                    if len(selected) >= args.total:
                        break
                    add_item(selected, selected_keys, selected_bucket_stats, item)
                if len(selected) >= args.total:
                    break

    candidates = selected

    logging.info("最终选取: %d 题", len(candidates))

    # 难度分布
    if student_accuracy:
        acc_bins = Counter()
        for c in candidates:
            acc = c.get('student_accuracy')
            if acc is None:
                acc_bins['未知'] += 1
            elif acc == 0:
                acc_bins['0% (全错)'] += 1
            elif acc < 0.25:
                acc_bins['1-24%'] += 1
            elif acc < 0.5:
                acc_bins['25-49%'] += 1
            elif acc < 0.75:
                acc_bins['50-74%'] += 1
            else:
                acc_bins['75-87%'] += 1
        logging.info("难度分布:")
        for k, v in sorted(acc_bins.items()):
            logging.info("  %s: %d", k, v)

    logging.info("采样桶分布:")
    for bucket in fill_order:
        if selected_bucket_stats[bucket]:
            logging.info("  %s: %d", bucket, selected_bucket_stats[bucket])

    # 4. 构造 verl parquet 格式
    random.shuffle(candidates)

    records = []
    for c in candidates:
        records.append({
            "data_source": "math",  # 路由到 verl 内置 math_dapo.compute_score
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                {"role": "user", "content": c['question']},
                {"role": "assistant", "content": ""},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": c['ground_truth'],
            },
            "extra_info": {
                "sample_id": make_sample_id(c['question'], c['ground_truth']),
                "category": c['category'],
                "student_accuracy": c.get('student_accuracy'),
                "teacher_status": c.get('teacher_status'),
                "difficulty_bucket": difficulty_bucket(c.get('student_accuracy'), c.get('teacher_status')),
            },
        })

    # 5. 拆分 train/test
    num_test = min(args.num_test, len(records))
    train_records = records[:len(records) - num_test]
    test_records = records[len(records) - num_test:]

    logging.info("Train: %d, Test: %d", len(train_records), len(test_records))

    # 6. 保存
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    logging.info("Saved %s (%d rows)", train_path, len(train_df))
    logging.info("Saved %s (%d rows)", test_path, len(test_df))

    # 验证
    verify = pd.read_parquet(train_path)
    s = verify.iloc[0]
    logging.info("=== 验证第一条 ===")
    logging.info("  data_source: %s", s['data_source'])
    logging.info("  prompt: %s", s['prompt'][0]['content'][:100])
    logging.info("  ground_truth: %s", s['reward_model']['ground_truth'])
    logging.info("  category: %s", s['extra_info'].get('category', 'N/A'))


if __name__ == "__main__":
    main()
