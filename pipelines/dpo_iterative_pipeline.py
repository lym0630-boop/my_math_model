"""
DPO 数据迭代构造 Pipeline (严格版)

策略:
  1. 答对率 >= 50% (4/8) 的题 → on-policy pair (真会做)
  2. 答对率 < 50% 但 > 0 → 丢弃 (不可靠)
  3. 全错 + Teacher 答对 → teacher-chosen pair
  4. 全错 + Teacher 也错 → 丢弃 (题太难)
  5. 全对 → 丢弃 (太简单)

迭代: 如果 pair 数不够 target，从候选池补采新题，重复以上流程

用法:
  阶段1 (离线，不需要 GPU):
    python3 dpo_iterative_pipeline.py --stage select --round 2
    → 从候选池选新题，输出 round2_questions.jsonl

  阶段2 (GPU 机器):
    python3 dpo_iterative_pipeline.py --stage student_infer --round 2
    → Student 推理

  阶段3 (GPU 机器):
    python3 dpo_iterative_pipeline.py --stage teacher_infer --round 2
    → Teacher 推理

  阶段4 (离线):
    python3 dpo_iterative_pipeline.py --stage assemble
    → 合并所有 round 数据，组装最终 DPO pairs
"""

import json
import re
import random
import argparse
import os
from collections import Counter
from pathlib import Path
import statistics


BASE_DIR = '/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline'
SFT_DIR = os.path.join(BASE_DIR, 'sft_data')

# ===================== 答案提取和校验 =====================

def extract_boxed_nested(text):
    """支持嵌套大括号的 \\boxed 提取"""
    results = []
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
    """将 LaTeX 表达式转数值"""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r'\\{1,2}(?:text|mathrm|textbf|mathbf)\{([^}]*)\}', r'\1', s)
    # \dfrac{a}{b}
    m = re.match(r'^(-?)\\{1,2}d?frac\{([^}]+)\}\{([^}]+)\}$', s)
    if m:
        try:
            sign = -1 if m.group(1) == '-' else 1
            return sign * float(m.group(2).strip()) / float(m.group(3).strip())
        except (ValueError, ZeroDivisionError):
            pass
    cleaned = s.replace(',', '').replace(' ', '').replace('\\,', '')
    try:
        return float(cleaned)
    except ValueError:
        pass
    m = re.match(r'^(-?\d+\.?\d*)\s*/\s*(\d+\.?\d*)$', cleaned)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def check_answer(pred_val, gt_val, tol=1e-3):
    """数值比较"""
    if pred_val is None or gt_val is None:
        return False
    if abs(gt_val) < 1e-10:
        return abs(pred_val) < tol
    if gt_val == int(gt_val) and abs(gt_val) < 1e9:
        return abs(pred_val - gt_val) < 0.5
    return (abs(pred_val - gt_val) / max(abs(gt_val), 1e-10) < tol
            or abs(pred_val - gt_val) < tol)


def extract_and_check(text, ground_truth):
    """提取答案并校验"""
    boxed_list = extract_boxed_nested(text)
    if not boxed_list:
        return None, False
    pred_latex = boxed_list[-1]
    pred_val = latex_to_number(pred_latex)
    gt_val = latex_to_number(ground_truth)
    return pred_latex, check_answer(pred_val, gt_val)


def remove_think_block(text):
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned
    if '</think>' in text:
        cleaned = text.split('</think>', 1)[1].strip()
        if cleaned:
            return cleaned
    return text


def ensure_think_block(text):
    """补齐官方 DeepSeek 模板中由 prompt 承载的开头 <think>。"""
    think_end = text.find('</think>')
    if think_end >= 0 and '<think>' not in text[:think_end]:
        return '<think>\n' + text.lstrip()
    return text


# ===================== 阶段1: 选题 =====================

def select_questions(round_num, num_questions, seed=42):
    """从候选池中选取新题（排除已用过的）"""
    random.seed(seed + round_num)

    # 收集已使用的题目
    used_keys = set()
    for r in range(1, round_num + 1):
        qfile = os.path.join(SFT_DIR, 'round%d_questions.jsonl' % r)
        if os.path.exists(qfile):
            with open(qfile) as f:
                for line in f:
                    d = json.loads(line)
                    used_keys.add(d['question'][:120])

    # round1 的就是原来的 12K
    if round_num == 1:
        r1_file = os.path.join(SFT_DIR, 'dpo_questions_12k.jsonl')
        if os.path.exists(r1_file):
            with open(r1_file) as f:
                for line in f:
                    d = json.loads(line)
                    used_keys.add(d['question'][:120])

    print("已使用 %d 道题" % len(used_keys))

    # 从完整候选池加载
    extracted_file = os.path.join(SFT_DIR, 'extracted_qa.jsonl')
    candidates = []

    with open(extracted_file) as f:
        for line in f:
            d = json.loads(line)
            # 和 select_dpo_questions.py 相同的硬性过滤
            if d['gt_source'] != 'code_output':
                continue
            gt = d['ground_truth']
            if gt is None:
                continue
            gt = gt.strip()
            if not (re.match(r'^-?\d+$', gt) or re.match(r'^-?\d+\.\d{1,6}$', gt)
                    or re.match(r'^-?\d+/\d+$', gt)):
                continue
            alen = len(d['answer'])
            qlen = len(d['question'])
            if not (50 <= qlen <= 1500 and 200 <= alen <= 800):
                continue

            qkey = d['question'][:120]
            if qkey in used_keys:
                continue

            # 排除物理/CS
            q_lower = d['question'][:500].lower()
            physics_kw = ['force', 'velocity', 'acceleration', 'newton', 'energy',
                          'momentum', 'circuit', 'voltage', 'resistance', 'kinetic',
                          'torque', 'magnetic', 'thermodynamic']
            cs_kw = ['algorithm', 'complexity', 'big-o', 'sorting', 'binary search',
                     'dynamic programming', 'graph algorithm', 'data structure']
            if any(w in q_lower for w in physics_kw + cs_kw):
                continue

            candidates.append(d)

    print("候选池剩余: %d 道" % len(candidates))

    # 随机采样
    random.shuffle(candidates)
    selected = candidates[:num_questions]

    # 写入
    outfile = os.path.join(SFT_DIR, 'round%d_questions.jsonl' % round_num)
    with open(outfile, 'w') as f:
        for d in selected:
            out = {
                'question': d['question'],
                'answer': d['answer'],
                'ground_truth': d['ground_truth'],
                'category': d.get('category', 'unknown'),
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

    print("选取 %d 道题 → %s" % (len(selected), outfile))
    return outfile


# ===================== 阶段2: Student 推理 =====================

def student_infer(round_num, model_path, tp_size=4, n_samples=8):
    """Student 批量推理"""
    from vllm import LLM, SamplingParams

    qfile = os.path.join(SFT_DIR, 'round%d_questions.jsonl' % round_num)
    outfile = os.path.join(SFT_DIR, 'round%d_student.jsonl' % round_num)

    data = []
    with open(qfile) as f:
        for line in f:
            data.append(json.loads(line))
    print("加载 %d 道题" % len(data))

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    prompts = []
    for d in data:
        prompt = (
            "<|im_start|>system\n"
            "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
            "<|im_start|>user\n" + d['question'] + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    import time
    start = time.time()
    # 分批推理
    batch_size = 2000
    all_outputs = []
    for bs in range(0, len(prompts), batch_size):
        be = min(bs + batch_size, len(prompts))
        print("  批次 %d~%d..." % (bs + 1, be))
        outputs = llm.generate(prompts[bs:be], sampling_params)
        all_outputs.extend(outputs)

    elapsed = time.time() - start
    print("推理完成: %.0fs" % elapsed)

    # 处理结果（使用修复后的提取逻辑）
    results = []
    for d, output in zip(data, all_outputs):
        gt = d['ground_truth']
        responses = []
        for comp in output.outputs:
            text = comp.text.strip()
            pred, is_correct = extract_and_check(text, gt)
            responses.append({
                'text': text,
                'predicted_answer': pred,
                'is_correct': is_correct,
            })

        results.append({
            'question': d['question'],
            'ground_truth': gt,
            'category': d.get('category', 'unknown'),
            'num_correct': sum(r['is_correct'] for r in responses),
            'num_total': len(responses),
            'responses': responses,
        })

    with open(outfile, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 统计
    all_c = sum(1 for r in results if r['num_correct'] == r['num_total'])
    half = sum(1 for r in results if r['num_correct'] >= r['num_total'] // 2 and r['num_correct'] < r['num_total'])
    below = sum(1 for r in results if 0 < r['num_correct'] < r['num_total'] // 2)
    zero = sum(1 for r in results if r['num_correct'] == 0)
    print("全对: %d, 答对>=半: %d, 答对<半: %d, 全错: %d" % (all_c, half, below, zero))
    print("输出: %s" % outfile)
    return outfile


# ===================== 阶段3: Teacher 推理 =====================

def teacher_infer(round_num, model_path, tp_size=4):
    """对全错题目用 Teacher 推理"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    student_file = os.path.join(SFT_DIR, 'round%d_student.jsonl' % round_num)
    outfile = os.path.join(SFT_DIR, 'round%d_teacher.jsonl' % round_num)

    # 只加载全错的
    allwrong = []
    with open(student_file) as f:
        for line in f:
            d = json.loads(line)
            if d['num_correct'] == 0:
                allwrong.append(d)
    print("全错题数: %d" % len(allwrong))

    if not allwrong:
        print("没有全错的题，跳过 Teacher 推理")
        with open(outfile, 'w') as f:
            pass
        return outfile

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=4096,
        stop=["<｜end▁of▁sentence｜>"],
    )

    prompts = []
    for d in allwrong:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": d['question']},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    import time
    start = time.time()
    batch_size = 500
    all_outputs = []
    for bs in range(0, len(prompts), batch_size):
        be = min(bs + batch_size, len(prompts))
        print("  批次 %d~%d..." % (bs + 1, be))
        outputs = llm.generate(prompts[bs:be], sampling_params)
        all_outputs.extend(outputs)

    elapsed = time.time() - start
    print("推理完成: %.0fs" % elapsed)

    results = []
    for d, output in zip(allwrong, all_outputs):
        text = ensure_think_block(output.outputs[0].text.strip())
        pred, is_correct = extract_and_check(text, d['ground_truth'])
        results.append({
            'question': d['question'],
            'ground_truth': d['ground_truth'],
            'category': d.get('category', 'unknown'),
            'teacher_response': remove_think_block(text),
            'teacher_full_response': text,
            'teacher_predicted': pred,
            'teacher_correct': is_correct,
        })

    with open(outfile, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    correct = sum(1 for r in results if r['teacher_correct'])
    print("Teacher 答对: %d / %d (%.1f%%)" % (correct, len(results), correct / len(results) * 100 if results else 0))
    print("输出: %s" % outfile)
    return outfile


# ===================== 阶段4: 合并组装 =====================

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def build_dpo_prompt(question):
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def assemble_all_rounds(max_round, target=8000, seed=42):
    """合并所有 round 的数据，组装 DPO pairs"""
    random.seed(seed)

    onpolicy_pairs = []
    teacher_pairs = []

    for rnd in range(1, max_round + 1):
        student_file = os.path.join(SFT_DIR, 'round%d_student.jsonl' % rnd)
        teacher_file = os.path.join(SFT_DIR, 'round%d_teacher.jsonl' % rnd)

        if not os.path.exists(student_file):
            continue

        # 加载 Teacher 结果
        teacher_map = {}
        if os.path.exists(teacher_file):
            with open(teacher_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    if t['teacher_correct']:
                        teacher_map[t['question'][:120]] = t

        # 加载 Student 结果
        with open(student_file) as f:
            for line in f:
                d = json.loads(line)
                nc = d['num_correct']
                nt = d['num_total']
                prompt = build_dpo_prompt(d['question'])

                if nc >= nt // 2 and nc < nt:
                    # ✅ 答对 >= 一半: on-policy pair
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
                            'category': d.get('category', 'unknown'),
                            'round': rnd,
                        })

                elif nc == 0:
                    # 全错: 看 Teacher
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
                            'category': d.get('category', 'unknown'),
                            'round': rnd,
                        })
                # nc < nt//2 and nc > 0 → 丢弃
                # nc == nt → 太简单丢弃

        print("Round %d: on-policy累计=%d, teacher累计=%d" % (
            rnd, len(onpolicy_pairs), len(teacher_pairs)))

    # 混合
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

    # 写入
    outfile = os.path.join(SFT_DIR, 'dpo_train_final.jsonl')
    with open(outfile, 'w') as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # 统计
    source_dist = Counter(p['source'] for p in final)
    round_dist = Counter(p['round'] for p in final)
    cat_dist = Counter(p['category'] for p in final)
    chosen_lens = [len(p['chosen']) for p in final]
    rejected_lens = [len(p['rejected']) for p in final]

    print("\n" + "=" * 60)
    print("最终 DPO 数据")
    print("=" * 60)
    print("总 pairs: %d / 目标 %d" % (len(final), target))
    enough = len(final) >= target * 0.9
    print("状态: %s" % ("✅ 达标" if enough else "❌ 不足，需要继续补采"))

    print("\n来源:")
    for src, cnt in source_dist.most_common():
        print("  %s: %d (%.1f%%)" % (src, cnt, cnt / len(final) * 100))

    print("\n各 round 贡献:")
    for rnd, cnt in sorted(round_dist.items()):
        print("  round %d: %d pairs" % (rnd, cnt))

    print("\n学科:")
    for cat, cnt in cat_dist.most_common():
        print("  %s: %d (%.1f%%)" % (cat, cnt, cnt / len(final) * 100))

    if chosen_lens:
        print("\nchosen 长度:  median=%d, mean=%d" % (
            statistics.median(chosen_lens), statistics.mean(chosen_lens)))
        print("rejected 长度: median=%d, mean=%d" % (
            statistics.median(rejected_lens), statistics.mean(rejected_lens)))

    print("\n输出: %s" % outfile)
    return len(final), enough


def main():
    parser = argparse.ArgumentParser(description='DPO 迭代构造 Pipeline')
    parser.add_argument('--stage', required=True,
                        choices=['select', 'student_infer', 'teacher_infer', 'assemble'],
                        help='执行哪个阶段')
    parser.add_argument('--round', type=int, default=1, help='当前是第几轮')
    parser.add_argument('--target', type=int, default=8000, help='目标 pair 数')
    parser.add_argument('--num_questions', type=int, default=16000, help='补采题数')
    parser.add_argument('--student_model',
        default=os.path.join(BASE_DIR, 'Qwen2.5-Math-7B-Instruct'))
    parser.add_argument('--teacher_model',
        default=os.path.join(BASE_DIR, 'DeepSeek-R1-Distill-Qwen-32B'))
    parser.add_argument('--tp_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.stage == 'select':
        print("===== 阶段1: 选题 (Round %d) =====" % args.round)
        select_questions(args.round, args.num_questions, args.seed)

    elif args.stage == 'student_infer':
        print("===== 阶段2: Student 推理 (Round %d) =====" % args.round)
        student_infer(args.round, args.student_model, args.tp_size)

    elif args.stage == 'teacher_infer':
        print("===== 阶段3: Teacher 推理 (Round %d) =====" % args.round)
        teacher_infer(args.round, args.teacher_model, args.tp_size)

    elif args.stage == 'assemble':
        print("===== 阶段4: 合并组装 (Round 1~%d) =====" % args.round)
        n_pairs, enough = assemble_all_rounds(args.round, args.target, args.seed)
        if not enough:
            print("\n⚠️  数据不足，建议进行 Round %d" % (args.round + 1))
            print("   python3 dpo_iterative_pipeline.py --stage select --round %d --num_questions 16000" % (args.round + 1))


if __name__ == '__main__':
    main()
