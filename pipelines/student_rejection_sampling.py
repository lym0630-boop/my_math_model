"""
Student Rejection Sampling: 用 Qwen2.5-Math-7B-Instruct 对 12K 题目各生成 8 个回答
用于后续 DPO 偏好对构造

运行环境: 4×A100-40GB
运行方式: python3 student_rejection_sampling.py

预计资源:
  - 模型: Qwen2.5-Math-7B BF16 ≈ 14GB 显存
  - vLLM TP=4, 每卡 ~3.5GB 模型权重 + KV cache, 非常宽裕
  - 12K × 8 = 96K 次生成, max_new_tokens=2048
  - 预计耗时: 约 1~2 小时
"""

import json
import re
import os
import time
import argparse
from pathlib import Path


def build_prompt(question):
    """
    构造 Qwen2.5-Math-Instruct 格式的 prompt
    模型的默认 system prompt: "Please reason step by step, and put your final answer within \\boxed{}."
    这正好适合我们的需求 —— 让模型用 \\boxed{} 给出最终答案，方便提取
    """
    return question


def extract_boxed_answer(text):
    """从模型输出中提取 \\boxed{} 中的答案"""
    # 匹配最后一个 \boxed{...}（模型可能输出多个中间结果）
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()

    # 备选: 匹配 "the answer is XXX" 模式
    m = re.search(r'(?:the\s+)?answer\s+is\s*[:\s]*([^\n\.]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def normalize_number(s):
    """将数值字符串标准化用于比较"""
    if s is None:
        return None
    s = s.strip().replace(',', '').replace(' ', '')

    # 分数 → float
    if '/' in s:
        try:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return None

    # 直接转 float
    try:
        return float(s)
    except ValueError:
        return None


def check_answer(predicted, ground_truth, tol=1e-4):
    """
    比较预测答案和 ground truth
    支持: 整数、小数、分数的数值比较，容忍小浮点误差
    """
    pred_val = normalize_number(predicted)
    gt_val = normalize_number(ground_truth)

    if pred_val is None or gt_val is None:
        return False

    # 对于整数 GT，要求精确匹配
    if gt_val == int(gt_val) and abs(gt_val) < 1e9:
        return abs(pred_val - gt_val) < 0.5  # 容忍四舍五入

    # 对于小数/分数，相对误差 < tol 或绝对误差 < tol
    if gt_val == 0:
        return abs(pred_val) < tol
    return abs(pred_val - gt_val) / max(abs(gt_val), 1e-10) < tol or abs(pred_val - gt_val) < tol


def run_vllm_inference(questions, model_path, n_samples=8, tp_size=4,
                       max_new_tokens=2048, temperature=0.7, top_p=0.9):
    """
    用 vLLM 批量推理
    对每道题生成 n_samples 个回答
    """
    from vllm import LLM, SamplingParams

    print(f"加载模型: {model_path}")
    print(f"  tensor_parallel_size={tp_size}")

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
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 构造 prompts（使用 chat template）
    # Qwen2.5-Math-Instruct 的格式:
    # <|im_start|>system\nPlease reason step by step, and put your final answer within \boxed{}.<|im_end|>
    # <|im_start|>user\n{question}<|im_end|>
    # <|im_start|>assistant\n
    prompts = []
    for q in questions:
        prompt = (
            "<|im_start|>system\n"
            "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{q}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    print(f"开始推理: {len(prompts)} 道题 × {n_samples} 个采样 = {len(prompts) * n_samples} 次生成")
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - start_time
    total_gen = len(prompts) * n_samples
    print(f"推理完成: 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min), {total_gen/elapsed:.1f} 样本/s")

    return outputs


def main():
    parser = argparse.ArgumentParser(description='Student Rejection Sampling for DPO')
    parser.add_argument(
        '--model_path',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/Qwen2.5-Math-7B-Instruct',
        help='Student 模型路径'
    )
    parser.add_argument(
        '--input',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/dpo_questions_12k.jsonl',
        help='筛选后的题目文件'
    )
    parser.add_argument(
        '--output',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/student_responses_12k.jsonl',
        help='输出文件路径'
    )
    parser.add_argument('--n_samples', type=int, default=8, help='每题生成几个回答')
    parser.add_argument('--tp_size', type=int, default=4, help='tensor parallel size')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    parser.add_argument('--batch_size', type=int, default=2000, help='分批推理大小（防OOM）')
    args = parser.parse_args()

    # 加载题目
    print("加载题目...")
    data = []
    with open(args.input) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"共 {len(data)} 道题")

    questions = [d['question'] for d in data]

    # 分批推理
    all_results = []
    for batch_start in range(0, len(questions), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]
        batch_data = data[batch_start:batch_end]

        print(f"\n===== 批次 {batch_start//args.batch_size + 1}: "
              f"题目 {batch_start+1}~{batch_end} =====")

        outputs = run_vllm_inference(
            batch_questions,
            args.model_path,
            n_samples=args.n_samples,
            tp_size=args.tp_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # 处理结果
        for idx, (d, output) in enumerate(zip(batch_data, outputs)):
            gt = d['ground_truth']
            responses = []

            for j, completion in enumerate(output.outputs):
                resp_text = completion.text.strip()
                predicted = extract_boxed_answer(resp_text)
                is_correct = check_answer(predicted, gt) if predicted else False

                responses.append({
                    'text': resp_text,
                    'predicted_answer': predicted,
                    'is_correct': is_correct,
                })

            correct_count = sum(r['is_correct'] for r in responses)

            result = {
                'question': d['question'],
                'ground_truth': gt,
                'category': d['category'],
                'num_correct': correct_count,
                'num_total': len(responses),
                'responses': responses,
            }
            all_results.append(result)

    # 写入结果
    print(f"\n写入 {args.output}...")
    with open(args.output, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # ========== 统计报告 ==========
    total_q = len(all_results)
    # 答对率分布
    acc_dist = {'全对': 0, '部分对': 0, '全错': 0}
    correct_rates = []
    for r in all_results:
        rate = r['num_correct'] / r['num_total']
        correct_rates.append(rate)
        if r['num_correct'] == r['num_total']:
            acc_dist['全对'] += 1
        elif r['num_correct'] == 0:
            acc_dist['全错'] += 1
        else:
            acc_dist['部分对'] += 1

    avg_rate = sum(correct_rates) / len(correct_rates) if correct_rates else 0

    print(f"\n{'='*60}")
    print(f"Student Rejection Sampling 统计")
    print(f"{'='*60}")
    print(f"总题数: {total_q}")
    print(f"每题采样数: {args.n_samples}")
    print(f"平均答对率: {avg_rate:.1%}")
    print(f"\n答对情况:")
    print(f"  全对 ({args.n_samples}/{args.n_samples}): {acc_dist['全对']:>6,} ({acc_dist['全对']/total_q*100:.1f}%) → 丢弃(无rejected)")
    print(f"  部分对:              {acc_dist['部分对']:>6,} ({acc_dist['部分对']/total_q*100:.1f}%) → ✅ 可构造 on-policy DPO pair")
    print(f"  全错 (0/{args.n_samples}):  {acc_dist['全错']:>6,} ({acc_dist['全错']/total_q*100:.1f}%) → 需 Teacher 补 chosen")
    print(f"\n预计可构造 DPO pairs:")
    print(f"  On-policy (策略A): ~{acc_dist['部分对']} 对")
    print(f"  Teacher补 (策略B): ~{acc_dist['全错']} 对 (需跑 Teacher)")
    print(f"  总计:              ~{acc_dist['部分对'] + acc_dist['全错']} 对")


if __name__ == '__main__':
    main()
