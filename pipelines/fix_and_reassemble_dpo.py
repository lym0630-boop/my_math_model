"""
修复 Teacher 答案提取 + 重新组装 DPO pairs

问题: 原始脚本的 \\boxed{} 提取有两个 bug:
  1. 反斜杠转义: JSON 中 \\boxed 需要 r'\\\\boxed' 来匹配
  2. 嵌套大括号: \\boxed{\\dfrac{1}{4}} 中有嵌套 {}，简单 [^}] 会截断

修复后 Teacher 准确率: 10.9% → 35.7% (救回 886 条)
"""

import json
import re
import random
import argparse
from collections import Counter
import statistics


# ===================== 修复后的答案提取 =====================

def extract_boxed_nested(text):
    """
    支持嵌套大括号的 \\boxed 提取
    适配 vLLM 输出中 \\\\boxed 的格式（JSON load 后双反斜杠）
    """
    results = []
    # 同时匹配 \\boxed 和 \boxed（兼容不同转义情况）
    for m in re.finditer(r'\\{1,2}boxed\s*\{', text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1])
    return results


def latex_to_number(s):
    """将 LaTeX 表达式转成数值，支持分数、文本包裹等"""
    if s is None:
        return None
    s = s.strip()

    # 移除 \\text{...}, \\mathrm{...}, \\textbf{...} 等包裹
    s = re.sub(r'\\{1,2}(?:text|mathrm|textbf|mathbf)\{([^}]*)\}', r'\1', s)

    # 匹配 \\dfrac{a}{b} 或 \\frac{a}{b}（可能有负号）
    m = re.match(r'^(-?)\\{1,2}d?frac\{([^}]+)\}\{([^}]+)\}$', s)
    if m:
        try:
            sign = -1 if m.group(1) == '-' else 1
            num = float(m.group(2).strip())
            den = float(m.group(3).strip())
            if den != 0:
                return sign * num / den
        except ValueError:
            pass

    # 纯数值（可能有逗号、空格）
    cleaned = s.replace(',', '').replace(' ', '').replace('\\,', '')
    try:
        return float(cleaned)
    except ValueError:
        pass

    # 分数 a/b
    m = re.match(r'^(-?\d+\.?\d*)\s*/\s*(\d+\.?\d*)$', cleaned)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass

    return None


def check_answer(pred_val, gt_val, tol=1e-3):
    """数值比较，容忍浮点误差"""
    if pred_val is None or gt_val is None:
        return False
    # 整数: 精确匹配（容忍四舍五入）
    if gt_val == int(gt_val) and abs(gt_val) < 1e9:
        return abs(pred_val - gt_val) < 0.5
    # 零
    if abs(gt_val) < 1e-10:
        return abs(pred_val) < tol
    # 相对误差或绝对误差
    return (abs(pred_val - gt_val) / max(abs(gt_val), 1e-10) < tol
            or abs(pred_val - gt_val) < tol)


def extract_and_check(text, ground_truth):
    """从文本中提取 boxed 答案并与 GT 比较"""
    boxed_list = extract_boxed_nested(text)
    if not boxed_list:
        return None, False

    pred_latex = boxed_list[-1]  # 取最后一个
    pred_val = latex_to_number(pred_latex)
    gt_val = latex_to_number(ground_truth)

    is_correct = check_answer(pred_val, gt_val)
    return pred_latex, is_correct


# ===================== 重新评估 Teacher =====================

def fix_teacher_responses(teacher_file, output_file):
    """用修复后的提取逻辑重新评估 Teacher 回答"""
    results = []
    original_correct = 0
    newly_correct = 0

    with open(teacher_file) as f:
        for line in f:
            d = json.loads(line)
            full = d['teacher_full_response']
            gt = d['ground_truth']

            if d['teacher_correct']:
                original_correct += 1
                # 原来就对的，保持不变
                results.append(d)
                continue

            # 原来判定为错，重新提取
            pred_latex, is_correct = extract_and_check(full, gt)

            if is_correct:
                newly_correct += 1
                # 更新字段
                d['teacher_predicted'] = pred_latex
                d['teacher_correct'] = True
                # 重新处理 response 文本（移除 think）
                d['teacher_response'] = remove_think_block(full)

            results.append(d)

    # 写入修复后的文件
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    total = len(results)
    total_correct = original_correct + newly_correct
    print("=" * 50)
    print("Teacher 答案修复")
    print("=" * 50)
    print("总题数:     %d" % total)
    print("原来答对:   %d" % original_correct)
    print("新救回:     %d" % newly_correct)
    print("修复后答对: %d (%.1f%%)" % (total_correct, total_correct / total * 100))
    print("写入: %s" % output_file)

    return results


def remove_think_block(text):
    """移除 <think>...</think>"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


# ===================== 重新评估 Student =====================

def fix_student_responses(student_file, output_file):
    """同样修复 Student 的答案提取"""
    results = []
    changes = 0

    with open(student_file) as f:
        for line in f:
            d = json.loads(line)
            gt = d['ground_truth']

            new_responses = []
            for r in d['responses']:
                # 重新提取和校验
                pred_latex, is_correct = extract_and_check(r['text'], gt)
                if is_correct != r['is_correct']:
                    changes += 1
                r['predicted_answer'] = pred_latex
                r['is_correct'] = is_correct
                new_responses.append(r)

            d['responses'] = new_responses
            d['num_correct'] = sum(1 for r in new_responses if r['is_correct'])
            results.append(d)

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 统计
    total = len(results)
    all_correct = sum(1 for r in results if r['num_correct'] == r['num_total'])
    partial = sum(1 for r in results if 0 < r['num_correct'] < r['num_total'])
    all_wrong = sum(1 for r in results if r['num_correct'] == 0)

    print("\n" + "=" * 50)
    print("Student 答案修复")
    print("=" * 50)
    print("判定变化数: %d" % changes)
    print("全对: %d (%.1f%%)" % (all_correct, all_correct / total * 100))
    print("部分对: %d (%.1f%%)" % (partial, partial / total * 100))
    print("全错: %d (%.1f%%)" % (all_wrong, all_wrong / total * 100))
    print("写入: %s" % output_file)

    return results


# ===================== 组装 DPO pairs =====================

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def build_dpo_prompt(question):
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT +
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + question +
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def assemble_pairs(student_data, teacher_data, target=8000, seed=42):
    """组装最终 DPO pairs"""
    random.seed(seed)

    # Teacher map
    teacher_map = {}
    for t in teacher_data:
        if t['teacher_correct']:
            teacher_map[t['question'][:120]] = t

    onpolicy_pairs = []
    teacher_pairs = []

    for d in student_data:
        nc = d['num_correct']
        nt = d['num_total']
        prompt = build_dpo_prompt(d['question'])

        if 0 < nc < nt:
            # 策略A: on-policy
            correct_resps = [r for r in d['responses'] if r['is_correct']]
            wrong_resps = [r for r in d['responses'] if not r['is_correct']]
            random.shuffle(correct_resps)
            random.shuffle(wrong_resps)

            n_pairs = 2 if (len(correct_resps) >= 2 and len(wrong_resps) >= 2) else 1
            for i in range(n_pairs):
                onpolicy_pairs.append({
                    'prompt': prompt,
                    'chosen': correct_resps[i]['text'],
                    'rejected': wrong_resps[i]['text'],
                    'source': 'on_policy',
                    'ground_truth': d['ground_truth'],
                    'category': d['category'],
                })

        elif nc == 0:
            # 策略B: teacher chosen
            qkey = d['question'][:120]
            if qkey in teacher_map:
                t = teacher_map[qkey]
                wrong_resps = [r for r in d['responses'] if not r['is_correct']]
                random.shuffle(wrong_resps)
                teacher_pairs.append({
                    'prompt': prompt,
                    'chosen': t['teacher_response'],
                    'rejected': wrong_resps[0]['text'],
                    'source': 'teacher_chosen',
                    'ground_truth': d['ground_truth'],
                    'category': d['category'],
                })

    # 混合: on-policy 全部 + teacher 补足
    total = len(onpolicy_pairs) + len(teacher_pairs)
    if total <= target:
        final = onpolicy_pairs + teacher_pairs
    else:
        if len(onpolicy_pairs) >= target:
            random.shuffle(onpolicy_pairs)
            final = onpolicy_pairs[:target]
        else:
            need = target - len(onpolicy_pairs)
            random.shuffle(teacher_pairs)
            final = onpolicy_pairs + teacher_pairs[:need]

    random.shuffle(final)
    return final, len(onpolicy_pairs), len(teacher_pairs)


def main():
    parser = argparse.ArgumentParser(description='修复答案提取 + 重新组装 DPO pairs')
    parser.add_argument('--student_file',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/student_responses_12k.jsonl')
    parser.add_argument('--teacher_file',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/teacher_responses.jsonl')
    parser.add_argument('--output',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/dpo_train_8k.jsonl')
    parser.add_argument('--target', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Step 1: 修复 Teacher
    teacher_fixed_file = args.teacher_file.replace('.jsonl', '_fixed.jsonl')
    teacher_data = fix_teacher_responses(args.teacher_file, teacher_fixed_file)

    # Step 2: 修复 Student
    student_fixed_file = args.student_file.replace('.jsonl', '_fixed.jsonl')
    student_data = fix_student_responses(args.student_file, student_fixed_file)

    # Step 3: 组装
    print("\n" + "=" * 50)
    print("组装 DPO pairs")
    print("=" * 50)

    final_pairs, n_onpolicy, n_teacher = assemble_pairs(
        student_data, teacher_data, args.target, args.seed)

    print("on-policy 可用: %d" % n_onpolicy)
    print("teacher 可用:   %d" % n_teacher)
    print("最终选取:       %d" % len(final_pairs))

    # 写入
    with open(args.output, 'w') as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # 统计
    source_dist = Counter(p['source'] for p in final_pairs)
    cat_dist = Counter(p['category'] for p in final_pairs)
    chosen_lens = [len(p['chosen']) for p in final_pairs]
    rejected_lens = [len(p['rejected']) for p in final_pairs]

    print("\n" + "=" * 60)
    print("最终 DPO 训练数据统计")
    print("=" * 60)
    print("总 pair 数: %d" % len(final_pairs))

    print("\n来源分布:")
    for src, cnt in source_dist.most_common():
        print("  %s: %d (%.1f%%)" % (src, cnt, cnt / len(final_pairs) * 100))

    print("\n学科分布:")
    for cat, cnt in cat_dist.most_common():
        print("  %s: %d (%.1f%%)" % (cat, cnt, cnt / len(final_pairs) * 100))

    print("\nchosen 长度:  median=%d, mean=%d" % (
        statistics.median(chosen_lens), statistics.mean(chosen_lens)))
    print("rejected 长度: median=%d, mean=%d" % (
        statistics.median(rejected_lens), statistics.mean(rejected_lens)))
    print("\n输出: %s" % args.output)


if __name__ == '__main__':
    main()
