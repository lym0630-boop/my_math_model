"""
GSM8K Few-Shot 评测脚本（适用于 base model）
=============================================

base model 没有 chat template，不能用 <|im_start|> 等标记。
正确做法是用 few-shot prompt：给几个示例，让模型续写。

评测协议：
  - 8-shot（与 Qwen2.5-Math 官方一致）
  - 示例从 train.jsonl 中取前 8 条
  - 答案提取：优先 \\boxed{}，其次 ####，最后取最后一个数字
  - 推理模式：greedy (temperature=0)

用法:
    python3 eval_gsm8k_fewshot.py \
        --model_path <模型路径> \
        --data_path ./gsm8k/test.jsonl \
        --train_path ./gsm8k/train.jsonl \
        --num_shots 8 \
        --tensor_parallel_size 4

对比 base 和 CPT 模型:
    python3 eval_gsm8k_fewshot.py --model_path ./Qwen2.5-Math-7B
    python3 eval_gsm8k_fewshot.py --model_path ./openwebmath/cpt_output/checkpoint-3800
"""

import json
import re
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K Few-Shot 评测")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--data_path", type=str,
                        default="/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/gsm8k/test.jsonl",
                        help="GSM8K test 数据路径")
    parser.add_argument("--train_path", type=str,
                        default="/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/gsm8k/train.jsonl",
                        help="GSM8K train 数据路径（取 few-shot 示例）")
    parser.add_argument("--output_path", type=str, default=None,
                        help="结果保存路径（jsonl）")
    parser.add_argument("--num_shots", type=int, default=8,
                        help="few-shot 示例数量（默认 8，设为 0 则为 0-shot）")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最多评测多少条，-1 表示全部")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="最大生成 token 数")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel 数量（GPU 数）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度，0 表示贪心")
    parser.add_argument("--gpu_util", type=float, default=0.72,
                        help="vLLM gpu_memory_utilization，few-shot 默认调低避免 OOM")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM 最大上下文长度")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="禁用 CUDA graph capture，降低初始化显存占用")
    parser.add_argument("--chat_template", action="store_true",
                        help="使用 Qwen2 chat template 格式（适合 SFT 后的模型）。"
                             "不加此参数则用纯文本续写格式（适合 base model）")
    parser.add_argument("--sft_format", action="store_true",
                        help="SFT 训练格式：system='You are Qwen...', user 加 'Solve...' 前缀。"
                             "需配合 --chat_template 使用")
    return parser.parse_args()


def load_jsonl(path, max_samples=-1):
    """加载 jsonl 文件"""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if max_samples > 0:
        data = data[:max_samples]
    return data


def extract_gold_answer(answer_str):
    """从 GSM8K 标准答案中提取数字（格式: #### 18）"""
    match = re.search(r"####\s*([\-\d\,\.]+)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def _round_float(s):
    """处理浮点垃圾：18.000000000000007 -> 18"""
    try:
        val = float(s)
        rounded = round(val)
        if abs(val - rounded) < 1e-4:
            return str(rounded)
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s
    except (ValueError, OverflowError):
        return s


def extract_pred_answer(pred_str):
    """
    从模型输出中提取答案
    核心策略：先在全文提取 boxed 和 ####，再截断废话做兜底
    """
    # 第 1 步：在全文找 \boxed{}（最可靠，取最后一个）
    idx = pred_str.rfind('\\boxed{')
    if idx != -1:
        depth = 0
        start = idx + len('\\boxed{')
        for i in range(start, len(pred_str)):
            if pred_str[i] == '{':
                depth += 1
            elif pred_str[i] == '}':
                if depth == 0:
                    ans = pred_str[start:i].strip()
                    ans = ans.replace(',', '').replace('$', '').strip()
                    nums = re.findall(r"[\-]?\d[\d,]*\.?\d*", ans)
                    if nums:
                        return _round_float(nums[-1].replace(",", ""))
                    return ans
                depth -= 1

    # 第 2 步：在全文找第一个 ####
    hash_match = re.search(r"####\s*([\-\d\,\.]+)", pred_str)
    if hash_match:
        return _round_float(hash_match.group(1).replace(",", "").strip())

    # 第 3 步：截断废话后兜底提取
    truncated = pred_str
    for sep in ['*Note:', '*note:', '(Note:', 'Note: The', '**Note',
                '*Verification', '*Context:', '*Correction']:
        pos = truncated.find(sep)
        if pos != -1:
            truncated = truncated[:pos]

    # **加粗** 数字
    bold = re.findall(r"\*\*\$?([\-\d,\.]+)\$?\*\*", truncated)
    if bold:
        ans = bold[-1].replace(",", "")
        if len(ans) <= 15:
            return _round_float(ans)

    # "= 数字"（取最后一个）
    eq_matches = re.findall(r"=\s*\$?([\-\d,\.]+)", truncated)
    if eq_matches:
        ans = eq_matches[-1].replace(",", "")
        if len(ans) <= 15:
            return _round_float(ans)

    # "answer is 数字"
    m = re.search(
        r"(?:answer|total|result|profit|cost|makes?|spend|earns?)\s+(?:is|are|was|=|:)\s+\$?([\-\d,\.]+)",
        truncated, re.IGNORECASE)
    if m:
        return _round_float(m.group(1).replace(",", ""))

    # 兜底：截断后最后一个合理数字
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", truncated)
    numbers = [n.replace(",", "") for n in numbers if len(n.replace(",", "")) <= 15]
    if numbers:
        return _round_float(numbers[-1])

    return None


def normalize_answer(ans):
    """标准化答案用于比较"""
    if ans is None:
        return None
    ans = ans.strip().rstrip(".")
    # 去掉末尾的 .0 和 .00
    if "." in ans:
        ans = ans.rstrip("0").rstrip(".")
    return ans


def clean_answer_annotations(answer_str):
    """
    清理 GSM8K answer 中的 <<...>> 计算标注
    例: 'Natalia sold 48/2 = <<48/2=24>>24 clips' -> 'Natalia sold 48/2 = 24 clips'
    这些标注是数据集自带的，base model 没见过，必须去掉
    """
    return re.sub(r"<<[^>]+>>", "", answer_str)


def build_fewshot_prompt(train_data, num_shots, question):
    """
    构建 few-shot prompt（纯文本续写格式，适合 base model）

    格式:
        Question: ...
        Answer: ...推理过程...
        #### 数字

        Question: ...
        Answer:
    """
    prompt = ""

    # few-shot 示例
    for i in range(min(num_shots, len(train_data))):
        item = train_data[i]
        q = item["question"].strip()
        a = clean_answer_annotations(item["answer"].strip())
        prompt += f"Question: {q}\nAnswer: {a}\n\n"

    # 当前问题
    prompt += f"Question: {question.strip()}\nAnswer:"

    return prompt


def build_chat_fewshot_prompt(train_data, num_shots, question, sft_format=False):
    """
    构建 chat template + few-shot prompt（适合 SFT 后的模型）

    把 few-shot 示例包成多轮对话：
        <|im_start|>system
        Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>
        <|im_start|>user
        示例题目1<|im_end|>
        <|im_start|>assistant
        示例解答1<|im_end|>
        <|im_start|>user
        示例题目2<|im_end|>
        <|im_start|>assistant
        示例解答2<|im_end|>
        ...
        <|im_start|>user
        真正要评测的题目<|im_end|>
        <|im_start|>assistant

    sft_format=True 时用和 SFT 训练一致的 system prompt 和 user 前缀
    """
    if sft_format:
        # 和训练格式对齐，并明确告诉模型题目是正确的，不要质疑
        system = ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
                  "All problems provided are verified and correct. "
                  "Solve them directly and put your final answer within \\boxed{}. "
                  "Do not question or doubt the problem statement.")
        user_prefix = "Solve the following mathematical problem.\n"
    else:
        system = "Please reason step by step, and put your final answer within \\boxed{}."
        user_prefix = ""

    # system message
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"

    # few-shot 示例作为多轮对话历史
    for i in range(min(num_shots, len(train_data))):
        item = train_data[i]
        q = item["question"].strip()
        a = clean_answer_annotations(item["answer"].strip())
        prompt += f"<|im_start|>user\n{user_prefix}{q}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{a}<|im_end|>\n"

    # 当前问题
    prompt += f"<|im_start|>user\n{user_prefix}{question.strip()}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    return prompt


def main():
    args = parse_args()

    # 自动生成输出路径
    if args.output_path is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        output_dir = "/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/gsm8k_results"
        os.makedirs(output_dir, exist_ok=True)
        args.output_path = os.path.join(output_dir, f"fewshot_{model_name}.jsonl")

    print("=" * 60)
    print("GSM8K Few-Shot 评测")
    print("=" * 60)
    print(f"模型路径:     {args.model_path}")
    print(f"数据路径:     {args.data_path}")
    print(f"Few-shot 数:  {args.num_shots}")
    print(f"Chat template: {'是' if args.chat_template else '否（纯文本续写）'}")
    print(f"SFT 格式:     {'是' if args.sft_format else '否（官方格式）'}")
    print(f"结果路径:     {args.output_path}")
    print(f"Temperature:  {args.temperature}")
    print(f"GPU util:     {args.gpu_util}")
    print(f"Max model len:{args.max_model_len}")
    print(f"Eager mode:   {'是' if args.enforce_eager else '否'}")
    print("=" * 60)

    # 加载数据
    test_data = load_jsonl(args.data_path, args.max_samples)
    train_data = load_jsonl(args.train_path)
    print(f"测试数据: {len(test_data)} 条")
    print(f"训练数据: {len(train_data)} 条（取前 {args.num_shots} 条作为示例）")

    # 打印一个完整 prompt 示例
    sample_prompt = build_fewshot_prompt(train_data, args.num_shots, test_data[0]["question"])
    print(f"\n--- Prompt 示例（前 500 字符）---")
    print(sample_prompt[:500])
    print(f"...\n--- Prompt 总长度: {len(sample_prompt)} 字符 ---\n")

    # 初始化 vLLM
    print("正在加载模型（vLLM）...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=args.enforce_eager,
    )
    # 停止条件
    if args.chat_template:
        # chat template 模式：遇到 <|im_end|> 停止
        stop_words = ["<|im_end|>"]
    else:
        # 纯文本模式：遇到下一道题的 Question 停止
        stop_words = ["\nQuestion:", "\n\nQuestion"]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=stop_words,
    )

    # 构造所有 prompt
    print("构造 prompts...")
    prompts = []
    for item in test_data:
        if args.chat_template:
            prompt = build_chat_fewshot_prompt(train_data, args.num_shots, item["question"], sft_format=args.sft_format)
        else:
            prompt = build_fewshot_prompt(train_data, args.num_shots, item["question"])
        prompts.append(prompt)

    # 批量推理
    print("开始推理...")
    outputs = llm.generate(prompts, sampling_params)

    # 评测
    correct = 0
    total = len(test_data)
    results = []

    for i, (item, output) in enumerate(zip(test_data, outputs)):
        pred_text = output.outputs[0].text.strip()
        gold = normalize_answer(extract_gold_answer(item["answer"]))
        pred = normalize_answer(extract_pred_answer(pred_text))

        is_correct = (pred is not None) and (gold is not None) and (pred == gold)
        if is_correct:
            correct += 1

        results.append({
            "question": item["question"],
            "gold_answer": gold,
            "pred_answer": pred,
            "pred_text": pred_text,
            "correct": is_correct,
        })

        if (i + 1) % 200 == 0:
            print(f"进度: {i+1}/{total}, 当前准确率: {correct/(i+1)*100:.2f}%")

    accuracy = correct / total * 100

    print()
    print("=" * 60)
    print(f"评测结果")
    print("=" * 60)
    print(f"模型:     {args.model_path}")
    print(f"总题数:   {total}")
    print(f"正确数:   {correct}")
    print(f"准确率:   {accuracy:.2f}%")
    print(f"Few-shot: {args.num_shots}-shot")
    print("=" * 60)

    # 保存详细结果
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 保存汇总
    summary_path = args.output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "num_shots": args.num_shots,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果: {args.output_path}")
    print(f"汇总结果: {summary_path}")

    # 打印几个错误样例方便排查
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n--- 前 3 个错误样例 ---")
        for e in errors[:3]:
            print(f"Q: {e['question'][:100]}...")
            print(f"Gold: {e['gold_answer']}")
            print(f"Pred: {e['pred_answer']}")
            print(f"Output: {e['pred_text'][:200]}...")
            print()


if __name__ == "__main__":
    main()
