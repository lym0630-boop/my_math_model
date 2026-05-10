"""
Teacher 推理: 用 DeepSeek-R1-Distill-Qwen-32B 为 Student 全错的题目生成 chosen 回答

运行环境: 4×A100-40GB
模型: DeepSeek-R1-Distill-Qwen-32B BF16, TP=4, 每卡 ~16GB

DeepSeek-R1 的 chat template 格式:
  <｜begin▁of▁sentence｜>{system_prompt}<｜User｜>{question}<｜Assistant｜><think>\n

R1 系列模型会先输出 <think>...</think> 思考过程，再输出最终答案
"""

import json
import re
import time
import argparse


def extract_r1_answer(text):
    """
    从 DeepSeek-R1 的输出中提取最终答案
    R1 输出格式: <think>思考过程</think>\n\n最终答案
    最终答案中通常包含 \\boxed{}
    """
    # 先尝试提取 </think> 后面的部分
    think_end = text.find('</think>')
    if think_end >= 0:
        final_part = text[think_end + len('</think>'):]
    else:
        final_part = text

    # 从最终部分提取 \boxed{}
    matches = re.findall(r'\\boxed\{([^}]*)\}', final_part)
    if matches:
        return matches[-1].strip()

    # 如果最终部分没有 boxed，全文搜索（取最后一个）
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()

    return None


def normalize_number(s):
    """将数值字符串标准化用于比较"""
    if s is None:
        return None
    s = s.strip().replace(',', '').replace(' ', '')
    if '/' in s:
        try:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def check_answer(predicted, ground_truth, tol=1e-4):
    """比较预测答案和 ground truth"""
    pred_val = normalize_number(predicted)
    gt_val = normalize_number(ground_truth)
    if pred_val is None or gt_val is None:
        return False
    if gt_val == int(gt_val) and abs(gt_val) < 1e9:
        return abs(pred_val - gt_val) < 0.5
    if gt_val == 0:
        return abs(pred_val) < tol
    return abs(pred_val - gt_val) / max(abs(gt_val), 1e-10) < tol or abs(pred_val - gt_val) < tol


def remove_think_block(text):
    """
    移除 <think>...</think> 思考过程，只保留最终回答
    DPO 的 chosen 不需要暴露 R1 的内部思考链
    """
    # 移除 <think>...</think>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned
    # 使用 DeepSeek 官方 generation prompt 时，<think> 可能属于 prompt 而不在生成文本里。
    if '</think>' in text:
        cleaned = text.split('</think>', 1)[1].strip()
        if cleaned:
            return cleaned
    # 如果移除后为空（不太可能），返回原文
    return text


def ensure_think_block(text):
    """补齐官方 DeepSeek 模板中由 prompt 承载的开头 <think>。"""
    think_end = text.find('</think>')
    if think_end >= 0 and '<think>' not in text[:think_end]:
        return '<think>\n' + text.lstrip()
    return text


def main():
    parser = argparse.ArgumentParser(description='Teacher inference for DPO chosen')
    parser.add_argument(
        '--model_path',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/DeepSeek-R1-Distill-Qwen-32B',
    )
    parser.add_argument(
        '--input',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/student_responses_12k.jsonl',
    )
    parser.add_argument(
        '--output',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/teacher_responses.jsonl',
    )
    parser.add_argument('--tp_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='R1 推理链较长，给更多空间')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--keep_think', action='store_true',
                        help='是否保留 <think> 思考过程（默认移除）')
    args = parser.parse_args()

    # 加载全错的题目
    print("加载 Student 全错的题目...")
    allwrong_data = []
    with open(args.input) as f:
        for line in f:
            d = json.loads(line)
            if d['num_correct'] == 0:
                allwrong_data.append(d)
    print("全错题数: %d" % len(allwrong_data))

    # 启动 vLLM
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("加载 Teacher 模型: %s" % args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=8192,       # R1 推理链长，需要更大 context
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    # Teacher 用 greedy decoding（temperature=0），确保 chosen 质量最高
    sampling_params = SamplingParams(
        n=1,
        temperature=0,           # greedy，质量最高
        max_tokens=args.max_new_tokens,
        stop=["<｜end▁of▁sentence｜>"],
    )

    # 构造 prompts: 使用 Teacher tokenizer 自带 chat_template，严格对齐 DeepSeek-R1 格式。
    questions = [d['question'] for d in allwrong_data]
    prompts = []
    for q in questions:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # 分批推理
    all_results = []
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_data = allwrong_data[batch_start:batch_end]

        print("\n批次 %d: 题目 %d~%d" % (
            batch_start // args.batch_size + 1,
            batch_start + 1, batch_end))

        start_time = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        elapsed = time.time() - start_time
        print("  耗时 %.0fs (%.1f 题/s)" % (elapsed, len(batch_prompts) / elapsed))

        for d, output in zip(batch_data, outputs):
            resp_text = ensure_think_block(output.outputs[0].text.strip())
            predicted = extract_r1_answer(resp_text)
            is_correct = check_answer(predicted, d['ground_truth']) if predicted else False

            # 处理输出文本
            if args.keep_think:
                chosen_text = resp_text
            else:
                chosen_text = remove_think_block(resp_text)

            result = {
                'question': d['question'],
                'ground_truth': d['ground_truth'],
                'category': d['category'],
                'teacher_response': chosen_text,
                'teacher_predicted': predicted,
                'teacher_correct': is_correct,
                'teacher_full_response': resp_text,  # 保留完整输出备用
            }
            all_results.append(result)

    # 写入结果
    print("\n写入 %s..." % args.output)
    with open(args.output, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 统计
    total = len(all_results)
    correct = sum(1 for r in all_results if r['teacher_correct'])
    print("\n" + "=" * 50)
    print("Teacher 推理统计")
    print("=" * 50)
    print("总题数:     %d" % total)
    print("Teacher答对: %d (%.1f%%)" % (correct, correct / total * 100))
    print("Teacher答错: %d (%.1f%%)" % (total - correct, (total - correct) / total * 100))
    print("\nTeacher 答对的题 → 可用作 DPO chosen")
    print("Teacher 也答错的 → 丢弃（太难了）")


if __name__ == '__main__':
    main()
