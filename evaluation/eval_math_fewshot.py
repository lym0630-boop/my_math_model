"""
MATH 8-Shot 评测脚本（chat template 格式）

评测协议:
  - 8-shot: 从 MATH test 中选 8 道短题作示范（覆盖 7 个类型）
  - 示范题不参与评测
  - chat template 格式（适合 Instruct 模型）
  - 答案提取: \\boxed{}
  - greedy decoding (temperature=0)

用法:
    python3 eval_math_fewshot.py --model_path Qwen2.5-Math-7B-Instruct --tp 2
"""

import json
import re
import os
import argparse
import pandas as pd
from pathlib import Path


PIPELINE_DIR = Path("/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline")
DATA_DIR = PIPELINE_DIR / "evaldata"
FEWSHOT_FILE = DATA_DIR / "math_fewshot_examples.json"
FEWSHOT_IDX_FILE = DATA_DIR / "math_fewshot_indices.json"
MATH_PARQUET = DATA_DIR / "test-00000-of-00001 (2).parquet"


def extract_boxed_answer(text):
    """从 \\boxed{} 中提取答案，支持嵌套大括号"""
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None
    start = idx + len('\\boxed{')
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def normalize_math_answer(ans):
    """标准化 MATH 答案用于比较"""
    if ans is None:
        return None
    ans = ans.strip()
    # 去掉 \text{}, \mathrm{} 等
    ans = re.sub(r'\\(?:text|mathrm|textbf)\{([^}]*)\}', r'\1', ans)
    # 去掉 $ 和空格
    ans = ans.replace('$', '').replace(' ', '')
    # \\dfrac → \\frac
    ans = ans.replace('\\dfrac', '\\frac')
    # 去掉末尾句号
    ans = ans.rstrip('.')
    return ans


def math_equiv(pred, gold):
    """判断两个 MATH 答案是否等价"""
    if pred is None or gold is None:
        return False
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    if p == g:
        return True
    # 尝试数值比较
    try:
        pv = float(p.replace(',', ''))
        gv = float(g.replace(',', ''))
        if abs(pv - gv) < 1e-6:
            return True
        if gv != 0 and abs(pv - gv) / abs(gv) < 1e-4:
            return True
    except (ValueError, ZeroDivisionError):
        pass
    return False


def build_fewshot_prompt(fewshot_examples, question):
    """
    构建 8-shot chat template prompt

    <|im_start|>system
    Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>
    <|im_start|>user
    示例题目1<|im_end|>
    <|im_start|>assistant
    示例解答1<|im_end|>
    ...（8个示例）
    <|im_start|>user
    实际题目<|im_end|>
    <|im_start|>assistant
    """
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    prompt = "<|im_start|>system\n%s<|im_end|>\n" % system

    for ex in fewshot_examples:
        prompt += "<|im_start|>user\n%s<|im_end|>\n" % ex['problem'].strip()
        prompt += "<|im_start|>assistant\n%s<|im_end|>\n" % ex['solution'].strip()

    prompt += "<|im_start|>user\n%s<|im_end|>\n" % question.strip()
    prompt += "<|im_start|>assistant\n"

    return prompt


def main():
    parser = argparse.ArgumentParser(description="MATH 8-Shot 评测")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_util", type=float, default=0.72,
                        help="vLLM gpu_memory_utilization，few-shot 默认调低避免 OOM")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM 最大上下文长度")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="禁用 CUDA graph capture，降低初始化显存占用")
    parser.add_argument("--limit", type=int, default=None, help="限制评测条数（调试用）")
    args = parser.parse_args()

    # 自动输出路径
    if args.output_path is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_path = str(PIPELINE_DIR / "eval_results" / ("math_fewshot_%s.jsonl" % model_name))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("=" * 60)
    print("MATH 8-Shot 评测")
    print("=" * 60)
    print("模型:   %s" % args.model_path)
    print("输出:   %s" % args.output_path)
    print("TP:     %d" % args.tp)
    print("GPU util: %.2f" % args.gpu_util)
    print("Max len: %d" % args.max_model_len)
    print("Eager:  %s" % ("yes" if args.enforce_eager else "no"))
    print("=" * 60)

    # 加载 few-shot 示例
    with open(FEWSHOT_FILE) as f:
        fewshot_examples = json.load(f)
    print("Few-shot 示例: %d 条" % len(fewshot_examples))

    # 加载要排除的 index
    with open(FEWSHOT_IDX_FILE) as f:
        exclude_indices = set(json.load(f))

    # 加载 MATH 数据
    df = pd.read_parquet(MATH_PARQUET)
    data = []
    for idx, row in df.iterrows():
        if idx in exclude_indices:
            continue
        gold = extract_boxed_answer(row['solution'])
        data.append({
            'question': row['problem'],
            'gold_answer': gold,
            'full_solution': row['solution'],
            'level': row['level'],
            'type': row['type'],
        })

    if args.limit:
        data = data[:args.limit]
    print("评测题数: %d (排除 %d 道 few-shot 示例题)" % (len(data), len(exclude_indices)))

    # 构造 prompts
    prompts = [build_fewshot_prompt(fewshot_examples, item['question']) for item in data]
    print("Prompt 示例长度: %d 字符" % len(prompts[0]))

    # vLLM 推理
    from vllm import LLM, SamplingParams

    print("加载模型...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=args.enforce_eager,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>"],
    )

    print("开始推理...")
    outputs = llm.generate(prompts, sampling_params)

    # 评测
    correct = 0
    total = len(data)
    results = []
    level_stats = {}
    type_stats = {}

    for i, (item, output) in enumerate(zip(data, outputs)):
        pred_text = output.outputs[0].text.strip()
        pred_answer = extract_boxed_answer(pred_text)
        gold = item['gold_answer']
        is_correct = math_equiv(pred_answer, gold)

        if is_correct:
            correct += 1

        # 按 level 统计
        lv = item['level']
        if lv not in level_stats:
            level_stats[lv] = {'correct': 0, 'total': 0}
        level_stats[lv]['total'] += 1
        if is_correct:
            level_stats[lv]['correct'] += 1

        # 按 type 统计
        tp = item['type']
        if tp not in type_stats:
            type_stats[tp] = {'correct': 0, 'total': 0}
        type_stats[tp]['total'] += 1
        if is_correct:
            type_stats[tp]['correct'] += 1

        results.append({
            'question': item['question'],
            'gold_answer': gold,
            'pred_answer': pred_answer,
            'pred_text': pred_text,
            'correct': is_correct,
            'level': lv,
            'type': tp,
        })

        if (i + 1) % 500 == 0:
            print("  进度: %d/%d, 当前准确率: %.2f%%" % (i + 1, total, correct / (i + 1) * 100))

    accuracy = correct / total * 100

    # 打印结果
    print()
    print("=" * 60)
    print("MATH 8-Shot 评测结果")
    print("=" * 60)
    print("模型:     %s" % args.model_path)
    print("总题数:   %d" % total)
    print("正确数:   %d" % correct)
    print("准确率:   %.2f%%" % accuracy)

    print()
    print("按难度:")
    for lv in sorted(level_stats.keys()):
        s = level_stats[lv]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print("  %-10s %6.2f%%  (%d/%d)" % (lv, acc, s['correct'], s['total']))

    print()
    print("按类型:")
    for tp in sorted(type_stats.keys()):
        s = type_stats[tp]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print("  %-25s %6.2f%%  (%d/%d)" % (tp, acc, s['correct'], s['total']))

    # 保存详细结果
    with open(args.output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 保存汇总
    summary_path = args.output_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'num_shots': 8,
            'level_stats': level_stats,
            'type_stats': type_stats,
        }, f, ensure_ascii=False, indent=2)

    print()
    print("详细结果: %s" % args.output_path)
    print("汇总结果: %s" % summary_path)


if __name__ == "__main__":
    main()
