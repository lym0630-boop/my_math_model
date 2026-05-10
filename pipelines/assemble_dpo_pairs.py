"""
DPO 偏好对组装脚本

输入:
  1. student_responses_12k.jsonl  (Student 的 8 次采样结果)
  2. teacher_responses.jsonl       (Teacher 对全错题的补充回答)

输出:
  dpo_train_8k.jsonl  (最终 DPO 训练数据)

组装策略:
  - 策略A (on-policy): Student 有对有错 → chosen=Student正确回答, rejected=Student错误回答
    - 如果 correct>=2 且 wrong>=2, 产出 2 对 (不同的 chosen/rejected 组合)
    - 否则产出 1 对
  - 策略B (teacher-chosen): Student 全错 + Teacher 答对 → chosen=Teacher回答, rejected=Student错误回答
  - 最终从 on-policy 和 teacher-chosen 中混合采样，凑满目标数量
"""

import json
import re
import random
import argparse
from collections import Counter


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_dpo_prompt(question):
    """构造 DPO 训练用的 prompt（Qwen2.5-Math-Instruct 格式）"""
    # 注意: DPO 训练时 prompt 要和 Student 推理时一致
    prompt = (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT +
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + question +
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def assemble_onpolicy_pairs(student_data):
    """
    策略A: 从 Student 的部分对结果中组装 on-policy DPO pairs
    """
    pairs = []

    for d in student_data:
        nc = d['num_correct']
        nt = d['num_total']

        # 只处理部分对（有正确也有错误）
        if nc == 0 or nc == nt:
            continue

        correct_resps = [r for r in d['responses'] if r['is_correct']]
        wrong_resps = [r for r in d['responses'] if not r['is_correct']]

        random.shuffle(correct_resps)
        random.shuffle(wrong_resps)

        prompt = build_dpo_prompt(d['question'])

        if len(correct_resps) >= 2 and len(wrong_resps) >= 2:
            # 产出 2 对，用不同的 chosen/rejected
            for i in range(2):
                pairs.append({
                    'prompt': prompt,
                    'chosen': correct_resps[i]['text'],
                    'rejected': wrong_resps[i]['text'],
                    'source': 'on_policy',
                    'ground_truth': d['ground_truth'],
                    'category': d['category'],
                    'student_accuracy': nc / nt,
                })
        else:
            # 产出 1 对
            pairs.append({
                'prompt': prompt,
                'chosen': correct_resps[0]['text'],
                'rejected': wrong_resps[0]['text'],
                'source': 'on_policy',
                'ground_truth': d['ground_truth'],
                'category': d['category'],
                'student_accuracy': nc / nt,
            })

    return pairs


def assemble_teacher_pairs(student_data, teacher_data):
    """
    策略B: Teacher chosen + Student rejected
    只用 Teacher 确实答对的题
    """
    # 用 question 做 key 匹配
    teacher_map = {}
    for t in teacher_data:
        if t['teacher_correct']:
            teacher_map[t['question'][:120]] = t

    pairs = []
    for d in student_data:
        if d['num_correct'] > 0:
            continue  # 只处理全错的

        qkey = d['question'][:120]
        if qkey not in teacher_map:
            continue  # Teacher 也答错了，丢弃

        t = teacher_map[qkey]
        wrong_resps = [r for r in d['responses'] if not r['is_correct']]
        random.shuffle(wrong_resps)

        prompt = build_dpo_prompt(d['question'])

        pairs.append({
            'prompt': prompt,
            'chosen': t['teacher_response'],
            'rejected': wrong_resps[0]['text'],
            'source': 'teacher_chosen',
            'ground_truth': d['ground_truth'],
            'category': d['category'],
            'student_accuracy': 0.0,
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description='组装 DPO 偏好对')
    parser.add_argument(
        '--student_file',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/student_responses_12k.jsonl',
    )
    parser.add_argument(
        '--teacher_file',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/teacher_responses.jsonl',
    )
    parser.add_argument(
        '--output',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/dpo_train_8k.jsonl',
    )
    parser.add_argument('--target', type=int, default=8000, help='目标 DPO pair 数量')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载 Student 结果
    print("加载 Student 结果...")
    student_data = []
    with open(args.student_file) as f:
        for line in f:
            student_data.append(json.loads(line))
    print("  %d 道题" % len(student_data))

    # 加载 Teacher 结果
    print("加载 Teacher 结果...")
    teacher_data = []
    with open(args.teacher_file) as f:
        for line in f:
            teacher_data.append(json.loads(line))
    teacher_correct = sum(1 for t in teacher_data if t['teacher_correct'])
    print("  %d 条 (Teacher答对 %d)" % (len(teacher_data), teacher_correct))

    # 组装两种 pair
    print("\n组装 on-policy pairs (策略A)...")
    onpolicy_pairs = assemble_onpolicy_pairs(student_data)
    print("  产出: %d pairs" % len(onpolicy_pairs))

    print("组装 teacher-chosen pairs (策略B)...")
    teacher_pairs = assemble_teacher_pairs(student_data, teacher_data)
    print("  产出: %d pairs" % len(teacher_pairs))

    # 混合采样凑目标数量
    total_available = len(onpolicy_pairs) + len(teacher_pairs)
    print("\n可用总数: %d (on-policy %d + teacher %d)" % (
        total_available, len(onpolicy_pairs), len(teacher_pairs)))

    if total_available <= args.target:
        # 全部使用
        final_pairs = onpolicy_pairs + teacher_pairs
        print("数据不足目标 %d，使用全部 %d pairs" % (args.target, len(final_pairs)))
    else:
        # 优先使用全部 on-policy，然后用 teacher 补
        if len(onpolicy_pairs) >= args.target:
            random.shuffle(onpolicy_pairs)
            final_pairs = onpolicy_pairs[:args.target]
        else:
            need_teacher = args.target - len(onpolicy_pairs)
            random.shuffle(teacher_pairs)
            final_pairs = onpolicy_pairs + teacher_pairs[:need_teacher]

    random.shuffle(final_pairs)
    print("最终: %d pairs" % len(final_pairs))

    # 写入
    print("\n写入 %s..." % args.output)
    with open(args.output, 'w') as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # ========== 统计报告 ==========
    source_dist = Counter(p['source'] for p in final_pairs)
    cat_dist = Counter(p['category'] for p in final_pairs)

    chosen_lens = [len(p['chosen']) for p in final_pairs]
    rejected_lens = [len(p['rejected']) for p in final_pairs]

    print("\n" + "=" * 60)
    print("DPO 训练数据统计")
    print("=" * 60)
    print("总 pair 数: %d" % len(final_pairs))
    print("\n来源分布:")
    for src, cnt in source_dist.most_common():
        print("  %s: %d (%.1f%%)" % (src, cnt, cnt / len(final_pairs) * 100))

    print("\n学科分布:")
    for cat, cnt in cat_dist.most_common():
        print("  %s: %d (%.1f%%)" % (cat, cnt, cnt / len(final_pairs) * 100))

    import statistics
    print("\nchosen 长度:  median=%d, mean=%d" % (
        statistics.median(chosen_lens), statistics.mean(chosen_lens)))
    print("rejected 长度: median=%d, mean=%d" % (
        statistics.median(rejected_lens), statistics.mean(rejected_lens)))

    print("\n输出文件: %s" % args.output)


if __name__ == '__main__':
    main()
