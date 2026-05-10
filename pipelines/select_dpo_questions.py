"""
从 extracted_qa.jsonl 中筛选 12K 道高质量题目，用于 DPO 数据构造

=== 筛选依据 ===

1. 【答案可验证性】只保留 GT 来源为 code_output 且 GT 为纯数值/分数的题目
   - 原因: DPO 的核心在于判断 Student 回答的对错，必须能自动精确校验
   - code_output 是最可靠的来源（直接来自代码运行结果）
   - 纯数值/分数可以直接做 == 比较，零误判率
   - 排除描述性文本答案（"the force is 7N to the right"），因为字符串匹配不可靠

2. 【难度控制】优先按 Student 正确率分桶，长度分层作为次级约束
   - 原因: Student 多采样正确率更接近“当前模型的真实难度”
   - 若已有 Student 采样结果，则优先保留:
     - 0.5~1.0: on-policy 甜区，最容易构造稳定 preference pair
     - 0~0.5: 中高难度，Student 有明确错误信号
     - 0.0: hard case，保留一部分留给 Teacher rescue
   - 若暂时没有 Student 难度信息，则退回到长度近似难度
   - answer 长度仍限制在 200~2000，避免极短 trivial 题和极长脏样本

3. 【题目长度】限制在 50~1500 字符
   - 原因: 过短(<50)的题目通常信息不完整或过于trivial
   - 过长(>1500)通常包含大量背景描述、表格等，不适合 DPO 训练

4. 【数值类型偏好】优先整数和简单小数，控制大数比例
   - 原因: 整数/简单小数的校验最可靠（3 == 3）
   - 大数(>10000)和高精度小数的校验容易出问题（浮点精度、格式差异）
   - 策略: 大数题目不超过总量的 5%

5. 【去重】按题目前120字符去重
   - 原因: 约1.4%的重复，虽然不多但会浪费 DPO 的宝贵配额
   - 重复题对 DPO 完全没有额外价值（同一题的偏好信号是冗余的）

6. 【学科多样性】按学科类别分层采样
   - 原因: 防止某一类题（如"其他数学"占48%）过度主导 DPO 训练
   - base model 是数学模型，应该在各数学子领域均衡提升
   - 排除物理和CS/算法题（与数学推理核心能力关系较弱）
   - 具体配比:
     代数/函数:  25%  ← 数学最核心的基础
     其他数学:   20%  ← 覆盖广泛的数学主题
     几何:       12%
     概率统计:   10%
     数列级数:    8%
     微积分:      7%
     抽象代数:    6%
     线性代数:    4%
     组合数学:    4%
     数论:        4%

7. 【排除非数学学科】去掉物理和CS/算法题
   - 原因: Qwen2.5-Math-7B-Instruct 是纯数学模型
   - 物理题依赖领域知识，不是数学推理能力的瓶颈
   - CS/算法题的评估逻辑不同（正确性不仅看数值）
"""

import json
import re
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict


# ===================== 学科分类 =====================

MATH_CATEGORIES = {
    '代数/函数': [
        'equation', 'solve for', 'roots', 'polynomial', 'quadratic',
        'factori', 'inequality', 'function', 'logarithm', 'exponential',
        'absolute value', 'rational', 'irrational',
    ],
    '几何': [
        'triangle', 'circle', 'angle', 'polygon', 'area of', 'perimeter',
        'geometry', 'coordinate', 'distance between', 'volume', 'surface area',
        'cone', 'sphere', 'cylinder', 'rectangle', 'square',
    ],
    '概率统计': [
        'probability', 'random variable', 'distribution', 'expectation',
        'variance', 'bayes', 'binomial', 'poisson', 'normal distribution',
        'standard deviation', 'hypothesis', 'confidence interval', 'sample',
    ],
    '数列级数': [
        'sequence', 'series', 'converge', 'sum of', 'fibonacci', 'recurrence',
        'arithmetic progression', 'geometric progression', 'nth term',
        'telescoping', 'partial sum',
    ],
    '微积分': [
        'integral', 'derivative', 'limit', 'calculus', 'differentiat',
        'antiderivat', 'd/dx', 'dy/dx', 'definite integral', 'improper integral',
        'partial derivative', "l'hopital", 'taylor', 'maclaurin',
    ],
    '抽象代数': [
        'group', 'ring', 'field theory', 'homomorphism', 'isomorphism',
        'subgroup', 'abelian', 'cyclic group', 'normal subgroup', 'quotient group',
        'kernel', 'coset',
    ],
    '线性代数': [
        'matrix', 'determinant', 'eigenvalue', 'linear algebra', 'vector space',
        'rank of', 'linear transformation', 'eigenvector', 'diagonaliz',
        'inner product', 'orthogonal', 'null space',
    ],
    '组合数学': [
        'combinat', 'permutation', 'choose', 'how many ways', 'counting',
        'arrangement', 'selection', 'binomial coefficient', 'pigeonhole',
        'inclusion-exclusion',
    ],
    '数论': [
        'number theory', 'prime', 'divisib', 'modular arithmetic', 'gcd',
        'lcm', 'congruence', 'remainder', 'euler', 'fermat', 'diophantine',
    ],
}

# 排除的学科
EXCLUDE_CATEGORIES = {
    '物理': [
        'force', 'velocity', 'acceleration', 'newton', 'energy', 'momentum',
        'circuit', 'voltage', 'resistance', 'kinetic', 'potential energy',
        'torque', 'magnetic', 'electric field', 'wave', 'optics', 'thermodynamic',
    ],
    'CS/算法': [
        'algorithm', 'complexity', 'big-o', 'sorting', 'binary search',
        'dynamic programming', 'graph algorithm', 'data structure', 'hash',
        'linked list', 'binary tree', 'stack', 'queue',
    ],
}

# 学科采样配比（占总 12K 的比例）
CATEGORY_RATIO = {
    '代数/函数': 0.25,
    '其他数学': 0.20,
    '几何': 0.12,
    '概率统计': 0.10,
    '数列级数': 0.08,
    '微积分': 0.07,
    '抽象代数': 0.06,
    '线性代数': 0.04,
    '组合数学': 0.04,
    '数论': 0.04,
}


def classify_question(question):
    """将题目分类到学科"""
    q = question[:500].lower()

    # 先检查是否要排除
    for cat, keywords in EXCLUDE_CATEGORIES.items():
        if any(w in q for w in keywords):
            return cat  # 返回物理/CS，后续会被过滤

    # 再匹配数学学科
    for cat, keywords in MATH_CATEGORIES.items():
        if any(w in q for w in keywords):
            return cat

    return '其他数学'


def is_valid_gt(gt_str):
    """检查 GT 是否为可精确校验的数值/分数"""
    gt = gt_str.strip()
    # 纯整数
    if re.match(r'^-?\d+$', gt):
        return True
    # 小数（最多6位小数）
    if re.match(r'^-?\d+\.\d{1,6}$', gt):
        return True
    # 分数
    if re.match(r'^-?\d+/\d+$', gt):
        return True
    return False


def is_big_number(gt_str):
    """检查是否为大数（绝对值 > 10000）"""
    gt = gt_str.strip()
    try:
        if '/' in gt:
            return False
        val = float(gt)
        return abs(val) > 10000
    except ValueError:
        return False


def load_student_accuracy(student_responses_path):
    """加载 Student 多采样统计，返回 {qkey: accuracy}。"""
    if not student_responses_path or not Path(student_responses_path).exists():
        return {}

    student_accuracy = {}
    with open(student_responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total = d.get('num_total', 0)
            if total <= 0:
                continue
            qkey = d['question'][:120].strip()
            student_accuracy[qkey] = d['num_correct'] / total
    return student_accuracy


def difficulty_bucket(acc):
    """按 Student 正确率分桶；若未知则返回 unknown。"""
    if acc is None:
        return 'unknown'
    if acc >= 1.0:
        return 'all_correct'
    if acc >= 0.5:
        return 'on_policy'
    if acc > 0.0:
        return 'mid_hard'
    return 'hard_zero'


def filter_and_select(input_path, output_path, total_target=12000, seed=42, student_responses_path=None):
    """主筛选逻辑"""
    random.seed(seed)
    student_accuracy = load_student_accuracy(student_responses_path)
    use_student_accuracy = bool(student_accuracy)

    # ========== 第一遍：加载并过滤 ==========
    print("第一遍：加载数据并应用硬性过滤条件...")
    candidates_by_cat = defaultdict(list)
    seen_keys = set()

    filter_stats = Counter()

    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            filter_stats['总条数'] += 1

            # 条件1: 如果是完整 extracted_qa，则 GT 来源必须是 code_output。
            # 若输入已是 bootstrap 后的精简题池，则可能没有 gt_source 字段，直接视为已通过该步。
            gt_source = d.get('gt_source')
            if gt_source is not None and gt_source != 'code_output':
                filter_stats['排除: GT来源非code_output'] += 1
                continue

            gt = d.get('ground_truth')
            if gt is None:
                filter_stats['排除: 无GT'] += 1
                continue

            # 条件2: GT 必须是可验证的数值/分数
            if not is_valid_gt(gt):
                filter_stats['排除: GT非数值/分数'] += 1
                continue

            # 条件3: 解题过程长度 200~2000（扩展上限，覆盖长推理题）
            alen = len(d['answer'])
            if alen < 200:
                filter_stats['排除: 解题过程太短(<200)'] += 1
                continue
            if alen > 2000:
                filter_stats['排除: 解题过程太长(>2000)'] += 1
                continue

            # 条件4: 题目长度 50~1500
            qlen = len(d['question'])
            if qlen < 50:
                filter_stats['排除: 题目太短(<50)'] += 1
                continue
            if qlen > 1500:
                filter_stats['排除: 题目太长(>1500)'] += 1
                continue

            # 条件5: 去重
            qkey = d['question'][:120].strip()
            if qkey in seen_keys:
                filter_stats['排除: 重复题目'] += 1
                continue
            seen_keys.add(qkey)

            acc = student_accuracy.get(qkey)
            if use_student_accuracy and acc is not None and acc >= 1.0:
                filter_stats['排除: Student全对'] += 1
                continue

            # 分类
            cat = d.get('category') or classify_question(d['question'])

            # 条件7: 排除物理和CS
            if cat in EXCLUDE_CATEGORIES:
                filter_stats[f'排除: {cat}题'] += 1
                continue

            d['student_accuracy'] = acc
            filter_stats['通过筛选'] += 1
            candidates_by_cat[cat].append(d)

    # ========== 打印过滤统计 ==========
    print(f"\n{'='*60}")
    print(f"筛选统计:")
    print(f"{'='*60}")
    for key in ['总条数', '排除: GT来源非code_output', '排除: 无GT',
                 '排除: GT非数值/分数', '排除: 解题过程太短(<200)',
                 '排除: 解题过程太长(>2000)', '排除: 题目太短(<50)',
                 '排除: 题目太长(>1500)', '排除: 重复题目', '排除: Student全对',
                 '排除: 物理题', '排除: CS/算法题', '通过筛选']:
        if key in filter_stats:
            print(f"  {key:30s}: {filter_stats[key]:>10,}")

    print(f"\n各学科候选数量:")
    for cat, items in sorted(candidates_by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:10s}: {len(items):>8,}")

    if use_student_accuracy:
        diff_stats = Counter()
        for items in candidates_by_cat.values():
            for d in items:
                diff_stats[difficulty_bucket(d.get('student_accuracy'))] += 1
        print("\nStudent 难度桶统计:")
        for key in ['all_correct', 'on_policy', 'mid_hard', 'hard_zero', 'unknown']:
            if key in diff_stats:
                print(f"  {key:12s}: {diff_stats[key]:>8,}")

    # ========== 第二遍：按学科 × 难度分层采样 ==========
    stage_desc = "Student 难度桶 × 学科配比" if use_student_accuracy else "学科配比 × 长度分层"
    print(f"\n第二遍：按{stage_desc}采样 {total_target} 条...")
    # 长度分层: 短(200-600):中(600-1200):长(1200-2000) ≈ 55:35:10
    LENGTH_TIERS = [
        ('短(200-600)',  200,  600,  0.55),
        ('中(600-1200)', 600,  1200, 0.35),
        ('长(1200-2000)',1200, 2000, 0.10),
    ]
    DIFFICULTY_BUCKETS = [
        ('on_policy', 0.45),
        ('mid_hard', 0.35),
        ('hard_zero', 0.15),
        ('unknown', 0.05),
    ]

    selected = []
    selection_stats = {}

    # 计算每个学科的目标数量
    targets = {}
    for cat, ratio in CATEGORY_RATIO.items():
        targets[cat] = int(total_target * ratio)

    # 大数配额（全局不超过5%）
    big_num_limit = int(total_target * 0.05)
    big_num_count = 0

    for cat, target_n in targets.items():
        pool = candidates_by_cat.get(cat, [])
        random.shuffle(pool)

        cat_selected = []
        if use_student_accuracy:
            difficulty_pools = defaultdict(list)
            for d in pool:
                difficulty_pools[difficulty_bucket(d.get('student_accuracy'))].append(d)

            selected_keys_cat = set()
            bucket_selected_counts = Counter()
            for bucket_name, bucket_ratio in DIFFICULTY_BUCKETS:
                bucket_target = max(1, int(target_n * bucket_ratio))
                bucket_pool = difficulty_pools.get(bucket_name, [])
                random.shuffle(bucket_pool)

                for d in bucket_pool:
                    if len(cat_selected) >= target_n:
                        break
                    if bucket_selected_counts[bucket_name] >= bucket_target:
                        break

                    qkey = d['question'][:120].strip()
                    if qkey in selected_keys_cat:
                        continue
                    if is_big_number(d['ground_truth']):
                        if big_num_count >= big_num_limit:
                            continue
                        big_num_count += 1
                    cat_selected.append(d)
                    selected_keys_cat.add(qkey)
                    bucket_selected_counts[bucket_name] += 1

            # 如果某些难度桶不够，从剩余池中按 on-policy -> mid-hard -> hard-zero -> unknown 补
            if len(cat_selected) < target_n:
                for bucket_name, _ in DIFFICULTY_BUCKETS:
                    bucket_pool = difficulty_pools.get(bucket_name, [])
                    random.shuffle(bucket_pool)
                    for d in bucket_pool:
                        if len(cat_selected) >= target_n:
                            break
                        qkey = d['question'][:120].strip()
                        if qkey in selected_keys_cat:
                            continue
                        if is_big_number(d['ground_truth']) and big_num_count >= big_num_limit:
                            continue
                        cat_selected.append(d)
                        selected_keys_cat.add(qkey)
                        bucket_selected_counts[bucket_name] += 1
        else:
            # 按长度分桶
            tier_pools = {}
            for tier_name, lo, hi, _ in LENGTH_TIERS:
                tier_pools[tier_name] = [d for d in pool if lo <= len(d['answer']) < hi]

            for tier_name, lo, hi, tier_ratio in LENGTH_TIERS:
                tier_target = max(1, int(target_n * tier_ratio))
                tier_pool = tier_pools[tier_name]
                random.shuffle(tier_pool)

                for d in tier_pool:
                    if len(cat_selected) >= target_n:
                        break
                    if sum(1 for s in cat_selected if lo <= len(s['answer']) < hi) >= tier_target:
                        break

                    # 条件4: 控制大数比例
                    if is_big_number(d['ground_truth']):
                        if big_num_count >= big_num_limit:
                            continue
                        big_num_count += 1

                    cat_selected.append(d)

            # 如果某个长度层不够，从其他层补
            if len(cat_selected) < target_n:
                selected_keys_cat = set(d['question'][:120] for d in cat_selected)
                for d in pool:
                    if len(cat_selected) >= target_n:
                        break
                    qk = d['question'][:120]
                    if qk in selected_keys_cat:
                        continue
                    if is_big_number(d['ground_truth']) and big_num_count >= big_num_limit:
                        continue
                    cat_selected.append(d)
                    selected_keys_cat.add(qk)

        selected.extend(cat_selected)
        actual = len(cat_selected)
        selection_stats[cat] = {
            'target': target_n,
            'actual': actual,
            'pool': len(pool),
            'difficulty_dist': Counter(difficulty_bucket(d.get('student_accuracy')) for d in cat_selected),
        }

    # 如果某些类别不够，从其他类别补
    shortfall = total_target - len(selected)
    if shortfall > 0:
        print(f"\n部分类别不足，需补充 {shortfall} 条...")
        # 收集已选的 ID
        selected_keys = set(d['question'][:120] for d in selected)
        # 从候选最多的类别补
        for cat in sorted(candidates_by_cat.keys(),
                          key=lambda c: len(candidates_by_cat[c]), reverse=True):
            if shortfall <= 0:
                break
            pool = candidates_by_cat[cat]
            random.shuffle(pool)
            for d in pool:
                if shortfall <= 0:
                    break
                qkey = d['question'][:120]
                if qkey not in selected_keys:
                    if is_big_number(d['ground_truth']) and big_num_count >= big_num_limit:
                        continue
                    selected.append(d)
                    selected_keys.add(qkey)
                    shortfall -= 1
                    selection_stats[cat]['actual'] = selection_stats.get(cat, {}).get('actual', 0) + 1

    # 最终打乱
    random.shuffle(selected)

    # ========== 写入结果 ==========
    print(f"\n写入 {output_path}...")
    with open(output_path, 'w') as f:
        for d in selected:
            # 输出精简格式，只保留 DPO 需要的字段
            out = {
                'question': d['question'],
                'answer': d['answer'],
                'ground_truth': d['ground_truth'],
                'category': d.get('category') or classify_question(d['question']),
            }
            if d.get('student_accuracy') is not None:
                out['student_accuracy'] = d['student_accuracy']
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

    # ========== 最终报告 ==========
    print(f"\n{'='*60}")
    print(f"最终选取: {len(selected)} 条")
    print(f"{'='*60}")
    print(f"\n{'学科':<10s} {'目标':>6s} {'实际':>6s} {'候选池':>8s} {'采样率':>8s}")
    print(f"{'-'*42}")
    for cat in sorted(selection_stats.keys(),
                      key=lambda c: selection_stats[c]['actual'], reverse=True):
        s = selection_stats[cat]
        rate = s['actual'] / s['pool'] * 100 if s['pool'] > 0 else 0
        print(f"{cat:<10s} {s['target']:>6d} {s['actual']:>6d} {s['pool']:>8d} {rate:>7.1f}%")

    # GT 类型统计
    gt_int = sum(1 for d in selected if re.match(r'^-?\d+$', d['ground_truth'].strip()))
    gt_float = sum(1 for d in selected if re.match(r'^-?\d+\.\d+$', d['ground_truth'].strip()))
    gt_frac = sum(1 for d in selected if re.match(r'^-?\d+/\d+$', d['ground_truth'].strip()))
    gt_big = sum(1 for d in selected if is_big_number(d['ground_truth']))

    print(f"\nGT 数值类型:")
    print(f"  整数:     {gt_int:>6d}")
    print(f"  小数:     {gt_float:>6d}")
    print(f"  分数:     {gt_frac:>6d}")
    print(f"  大数(>1万): {gt_big:>5d}")

    # 长度分层统计
    print(f"\n答案长度分层:")
    for tier_name, lo, hi, tier_ratio in LENGTH_TIERS:
        cnt = sum(1 for d in selected if lo <= len(d['answer']) < hi)
        print(f"  {tier_name}: {cnt:>6d} ({cnt/len(selected)*100:.1f}%, 目标{tier_ratio*100:.0f}%)")

    if use_student_accuracy:
        selected_diff = Counter(difficulty_bucket(d.get('student_accuracy')) for d in selected)
        print("\n最终 Student 难度桶:")
        for key in ['on_policy', 'mid_hard', 'hard_zero', 'unknown']:
            if key in selected_diff:
                print(f"  {key:12s}: {selected_diff[key]:>6d} ({selected_diff[key]/len(selected)*100:.1f}%)")

    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为 DPO 筛选高质量数学题')
    parser.add_argument(
        '--input',
        default='/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/sft_data/extracted_qa.jsonl',
        help='extracted_qa.jsonl 路径'
    )
    parser.add_argument(
        '--output',
        default='/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/sft_data/dpo_questions_12k.jsonl',
        help='输出路径'
    )
    parser.add_argument(
        '--total', type=int, default=12000,
        help='总共选多少道题（默认12000）'
    )
    parser.add_argument(
        '--student_responses',
        default='/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/sft_data/student_responses_24k_v2.jsonl',
        help='Student 多采样结果路径；存在时按 student_accuracy 分桶采样'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子'
    )
    args = parser.parse_args()

    filter_and_select(args.input, args.output, args.total, args.seed, args.student_responses)
