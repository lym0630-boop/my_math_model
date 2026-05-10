"""
从 stage3-qa 数据中提取独立的 (question, answer, ground_truth) 三元组
用于后续 DPO 数据构造

输入: stage3-qa/*.jsonl  (每条含多个 Question/Answer/Code)
输出: extracted_qa.jsonl  (每条一个独立 QA 对)

格式:
{
    "question": "题目文本",
    "answer": "完整解题过程",
    "code": "代码实现（如有）",
    "ground_truth": "从 code output 或 answer 中提取的最终答案",
    "gt_source": "提取来源: code_output / boxed / answer_tail / none",
    "source_file": "来源文件名",
    "source_idx": 原始样本索引,
    "question_idx": 题目在样本中的编号
}
"""

import json
import re
import os
import argparse
from pathlib import Path
from collections import Counter


def find_markers(text):
    """找出所有 **Question N** / **Answer N** / **Code Implementation N** 标记的位置"""
    markers = []
    for m in re.finditer(
        r'\*\*(Question|Answer|Code Implementation) (\d+)\*\*:\s*\n?', text
    ):
        markers.append({
            'type': m.group(1),      # Question / Answer / Code Implementation
            'num': int(m.group(2)),   # 编号
            'start': m.start(),       # 标记起始位置
            'end': m.end(),           # 标记结束位置（内容开始位置）
        })
    return markers


def extract_questions_from_sample(text, source_file, source_idx):
    """从一条样本中提取所有独立的 QA 对"""
    markers = find_markers(text)
    if not markers:
        return []

    # 按编号分组
    grouped = {}
    for mk in markers:
        num = mk['num']
        if num not in grouped:
            grouped[num] = {}
        grouped[num][mk['type']] = mk

    results = []
    for qnum in sorted(grouped.keys()):
        group = grouped[qnum]

        # 必须至少有 Question 和 Answer
        if 'Question' not in group or 'Answer' not in group:
            continue

        q_mk = group['Question']
        a_mk = group['Answer']
        c_mk = group.get('Code Implementation')

        # --- 提取 Question 文本 ---
        # 从 Question 标记结束到 Answer 标记开始
        question_text = text[q_mk['end']:a_mk['start']].strip()

        # --- 提取 Answer 文本 ---
        # 从 Answer 标记结束到下一个标记开始
        if c_mk:
            answer_text = text[a_mk['end']:c_mk['start']].strip()
        else:
            # 没有 Code，找下一个 Question 或文本结尾
            next_start = _find_next_section_start(markers, a_mk, text)
            answer_text = text[a_mk['end']:next_start].strip()

        # --- 提取 Code 文本 ---
        code_text = ""
        if c_mk:
            next_start = _find_next_section_start(markers, c_mk, text)
            code_text = text[c_mk['end']:next_start].strip()

        # --- 提取 ground truth ---
        ground_truth, gt_source = extract_ground_truth(answer_text, code_text)

        # 基本质量过滤：题目和答案都不能太短
        if len(question_text) < 10 or len(answer_text) < 20:
            continue

        results.append({
            'question': question_text,
            'answer': answer_text,
            'code': code_text,
            'ground_truth': ground_truth,
            'gt_source': gt_source,
            'source_file': source_file,
            'source_idx': source_idx,
            'question_idx': qnum,
        })

    return results


def _find_next_section_start(markers, current_mk, text):
    """找到当前 marker 之后的下一个 marker 的起始位置，或文本结尾"""
    next_starts = [
        mk['start'] for mk in markers
        if mk['start'] > current_mk['start'] and mk is not current_mk
    ]
    return min(next_starts) if next_starts else len(text)


def extract_ground_truth(answer_text, code_text):
    """
    按优先级从多个来源提取 ground truth
    优先级: code_output > boxed > answer_tail
    """
    gt, source = None, 'none'

    # 1. 从 Code 的 # Output: 注释中提取
    if code_text:
        outputs = re.findall(r'#\s*[Oo]utput:\s*(.+)', code_text)
        if outputs:
            # 取最后一个 output（通常是最终结果）
            gt = outputs[-1].strip()
            source = 'code_output'
            return gt, source

    # 2. 从 Answer 中提取 \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', answer_text)
    if boxed:
        gt = boxed[-1].strip()
        source = 'boxed'
        return gt, source

    # 3. 从 Answer 尾部提取（"答案为"、"answer is" 等）
    # 3a. 中文模式："答案为 XXX"
    zh_match = re.search(r'答案[为是]\s*[：:\s]*(.+?)$', answer_text, re.MULTILINE)
    if zh_match:
        gt = zh_match.group(1).strip().rstrip('。.')
        source = 'answer_tail_zh'
        return gt, source

    # 3b. 英文模式："the answer is XXX"
    en_match = re.search(
        r'(?:the\s+)?answer\s+is\s*[：:\s]*(.+?)\.?\s*$',
        answer_text, re.IGNORECASE | re.MULTILINE
    )
    if en_match:
        gt = en_match.group(1).strip().rstrip('.')
        source = 'answer_tail_en'
        return gt, source

    # 3c. 从 Answer 最后一个 $...$ 或 **...** 提取（加粗/公式通常是最终答案）
    bold_match = re.findall(r'\*\*([^*]+)\*\*', answer_text[-200:])
    if bold_match:
        candidate = bold_match[-1].strip()
        # 过滤掉太长的（不像答案）或者是标题
        if len(candidate) < 80 and not candidate.startswith(('Question', 'Answer', 'Code')):
            gt = candidate
            source = 'answer_bold'
            return gt, source

    return gt, source


def process_all_files(input_dir, output_path, max_samples=None):
    """处理所有 jsonl 文件"""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob('*.jsonl'))

    print(f"找到 {len(files)} 个文件")

    all_results = []
    stats = Counter()

    for fpath in files:
        fname = fpath.name
        print(f"\n处理: {fname}")
        file_count = 0

        with open(fpath) as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    stats['json_error'] += 1
                    continue

                text = d.get('text', '')
                if not text:
                    stats['empty_text'] += 1
                    continue

                qas = extract_questions_from_sample(text, fname, idx)
                for qa in qas:
                    all_results.append(qa)
                    stats[f'gt_{qa["gt_source"]}'] += 1
                    file_count += 1

                stats['total_samples'] += 1

        print(f"  提取 {file_count} 条 QA 对")

    # 写入输出文件
    print(f"\n=== 写入 {output_path} ===")
    with open(output_path, 'w') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 统计报告
    total_qa = len(all_results)
    has_gt = sum(1 for r in all_results if r['ground_truth'] is not None)

    print(f"\n{'='*50}")
    print(f"总处理样本数: {stats['total_samples']}")
    print(f"提取 QA 对数:  {total_qa}")
    print(f"有 ground truth: {has_gt} ({has_gt/total_qa*100:.1f}%)")
    print(f"无 ground truth: {total_qa - has_gt} ({(total_qa-has_gt)/total_qa*100:.1f}%)")
    print(f"\nGround truth 来源分布:")
    for key in sorted(stats.keys()):
        if key.startswith('gt_'):
            print(f"  {key[3:]:20s}: {stats[key]:>8d}")
    if stats.get('json_error'):
        print(f"\nJSON 解析错误: {stats['json_error']}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从 stage3-qa 提取独立 QA 对')
    parser.add_argument(
        '--input_dir',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/traindata/swallowmath_qa/stage3-qa',
        help='stage3-qa 数据目录'
    )
    parser.add_argument(
        '--output',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/extracted_qa.jsonl',
        help='输出文件路径'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='每个文件最多处理多少条样本（调试用）'
    )
    args = parser.parse_args()

    process_all_files(args.input_dir, args.output, args.max_samples)
