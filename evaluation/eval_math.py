"""
数学能力评测脚本 - 支持 GSM8K 和 MATH 数据集
用法:
    # 评测 GSM8K
    python eval_math.py --model /path/to/model --dataset gsm8k
    # 评测 MATH
    python eval_math.py --model /path/to/model --dataset math
    # 同时评测两个
    python eval_math.py --model /path/to/model --dataset all
    # 通过 vLLM API 评测（需先启动 vLLM 服务）
    python eval_math.py --api_url http://localhost:8000/v1 --model_name Qwen2.5-Math-7B --dataset all
"""

import argparse
import json
import math
import re
import os
import time
import pandas as pd
from pathlib import Path
from fractions import Fraction


# ==================== 答案提取 ====================

def extract_gsm8k_answer(text: str) -> str:
    """从 GSM8K 格式中提取最终答案，先截断废话再提取"""
    # 截断 *Note:* 等废话
    for sep in ['*Note:', '*note:', '(Note:', '*Verification', '*Context:', '*Correction']:
        pos = text.find(sep)
        if pos != -1:
            text = text[:pos]

    # 1. \boxed{}
    idx = text.rfind('\\boxed{')
    if idx != -1:
        depth = 0
        start = idx + len('\\boxed{')
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                if depth == 0:
                    ans = text[start:i].strip().replace(',', '').replace('$', '').strip()
                    nums = re.findall(r'[\-]?\d[\d,]*\.?\d*', ans)
                    if nums: return nums[-1].replace(',', '')
                    return ans
                depth -= 1

    # 2. ####
    patterns = [
        r'####\s*([\-\d,\.]+)',
        r'answer is\s*\$?([\-\d,\.]+)',
        r'answer:\s*\$?([\-\d,\.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '').strip()

    # 3. **加粗** 数字
    bold = re.findall(r'\*\*\$?([\-\d,\.]+)\$?\*\*', text)
    if bold:
        return bold[-1].replace(',', '').strip()

    # 4. 兜底：最后一个数字
    numbers = re.findall(r'[\-\d,\.]+', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return ""


def extract_math_answer(text: str) -> str:
    """从 MATH 格式中提取答案，先截断废话再提取"""
    # 截断 *Note:* 等废话
    for sep in ['*Note:', '*note:', '(Note:', '*Verification', '*Context:', '*Correction']:
        pos = text.find(sep)
        if pos != -1:
            text = text[:pos]

    # \boxed{...}（处理嵌套花括号）
    idx = text.rfind('\\boxed{')
    if idx != -1:
        depth = 0
        start = idx + len('\\boxed{')
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1

    # 备选
    patterns = [
        r'answer is\s*\$?(.*?)\$?\s*[.\n]',
        r'answer:\s*\$?(.*?)\$?\s*[.\n]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较"""
    answer = answer.strip()
    if not answer:
        return ""

    # 去掉常见 LaTeX 展示噪声
    answer = answer.replace('$', '').replace('%', '')
    answer = answer.replace('\\left', '').replace('\\right', '')
    answer = answer.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    answer = answer.replace('\\!', '').replace('\\,', '').replace('\\;', '').replace('\\:', '')

    # 去掉尾部句号/分号等，不动逗号（逗号可能是答案本身）
    answer = answer.strip().rstrip('.;')

    # 区间题常见写法：x \in [-2,7] -> [-2,7]
    answer = re.sub(r'^\s*[a-zA-Z]\s*\\in\s*', '', answer)

    # 标准化逗号和并集周围空格
    answer = re.sub(r'\s*,\s*', ',', answer)
    answer = re.sub(r'\s*\\cup\s*', r'\\cup', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    # 单个数值中的千分位逗号不影响比较；列表中的逗号要保留
    if re.fullmatch(r'[-+]?\d[\d,]*(?:\.\d+)?', answer):
        answer = answer.replace(',', '')

    # 先尝试转为数字做比较
    try:
        val = float(answer)
        if not math.isfinite(val):
            return answer.lower().strip()
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        pass

    # 规范化简单分数
    frac_match = re.fullmatch(r'([-+]?)\s*(\d+)\s*/\s*(\d+)', answer)
    if frac_match:
        sign = -1 if frac_match.group(1) == '-' else 1
        numerator = int(frac_match.group(2))
        denominator = int(frac_match.group(3))
        if denominator != 0:
            frac = sign * Fraction(numerator, denominator)
            return f"{frac.numerator}/{frac.denominator}"

    latex_frac_match = re.fullmatch(r'([-+]?)\\frac\{(-?\d+)\}\{(-?\d+)\}', answer)
    if latex_frac_match:
        sign = -1 if latex_frac_match.group(1) == '-' else 1
        numerator = int(latex_frac_match.group(2))
        denominator = int(latex_frac_match.group(3))
        if denominator != 0:
            frac = sign * Fraction(numerator, denominator)
            return f"{frac.numerator}/{frac.denominator}"

    # 对符号表达式，去掉所有空白和大小写差异，覆盖：
    # \frac{x + 2}{7} vs \frac{x+2}{7}
    # 6 + 9i vs 6+9i
    # x \in [-2,7] vs [-2, 7]
    return re.sub(r'\s+', '', answer).lower()


def parse_numeric_answer(answer: str):
    """尽量把答案解析成数值；失败时返回 None。"""
    if not answer:
        return None

    try:
        val = float(answer)
        if math.isfinite(val):
            return val
    except (ValueError, OverflowError):
        pass

    frac_match = re.fullmatch(r'([-+]?)(\d+)/(\d+)', answer)
    if frac_match:
        sign = -1 if frac_match.group(1) == '-' else 1
        numerator = int(frac_match.group(2))
        denominator = int(frac_match.group(3))
        if denominator != 0:
            return float(sign * Fraction(numerator, denominator))

    return None


def check_answer(pred: str, gold: str) -> bool:
    """比较预测答案和标准答案"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    # 数值比较（处理浮点误差）
    pred_num = parse_numeric_answer(pred_norm)
    gold_num = parse_numeric_answer(gold_norm)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    return False


# ==================== 数据加载 ====================

DATA_DIR = Path(__file__).parent / "evaldata"
GSM8K_JSONL = Path(__file__).parent / "gsm8k" / "test.jsonl"

def load_gsm8k():
    """加载 GSM8K 数据（优先用 jsonl，里面有标准的 #### 格式答案）"""
    data = []
    if GSM8K_JSONL.exists():
        with open(GSM8K_JSONL, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gold = extract_gsm8k_answer(row['answer'])
                data.append({
                    'question': row['question'],
                    'gold_answer': gold,
                    'full_solution': row['answer'],
                })
    else:
        # 兜底：用 parquet（注意这个数据源的 answer 格式不同）
        df = pd.read_parquet(DATA_DIR / "test-00000-of-00001 (1).parquet")
        for _, row in df.iterrows():
            gold = extract_gsm8k_answer(row['answer'])
            data.append({
                'question': row['question'],
                'gold_answer': gold,
                'full_solution': row['answer'],
            })
    return data


def load_math():
    """加载 MATH 数据"""
    df = pd.read_parquet(DATA_DIR / "test-00000-of-00001 (2).parquet")
    data = []
    for _, row in df.iterrows():
        gold = extract_math_answer(row['solution'])
        data.append({
            'question': row['problem'],
            'gold_answer': gold,
            'full_solution': row['solution'],
            'level': row['level'],
            'type': row['type'],
        })
    return data


# ==================== Prompt 构建（官方格式） ====================

# Qwen2.5-Math 官方 system prompt
SYSTEM_PROMPT_COT = "Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_PROMPT_TIR = "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."


def build_messages(question: str, mode: str = "cot") -> list:
    """
    构建 Qwen2.5-Math 官方 chat 格式的 messages。
    官方要求使用 chat template，不能直接拼字符串。
    注意：GSM8K 和 MATH 都统一用 \\boxed{} 格式输出答案。

    Args:
        question: 题目内容
        mode: "cot"（Chain-of-Thought）或 "tir"（Tool-integrated Reasoning）
    """
    system_prompt = SYSTEM_PROMPT_COT if mode == "cot" else SYSTEM_PROMPT_TIR
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def messages_to_prompt(messages: list) -> str:
    """
    手动拼 Qwen2 chat template 字符串。
    vLLM tokenizer 会自动将 <|im_start|> 等识别为 special token。
    """
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def build_prompt_gsm8k(question: str, use_sft_format: bool = False) -> str:
    """构建 GSM8K 的 chat template prompt"""
    if use_sft_format:
        system = ("Please reason step by step, and put your final answer within \\boxed{}. "
                  "All problems are verified and correct. "
                  "Do not question, doubt, or add notes about the problem statement. Focus only on solving.")
        user = question
    else:
        system = "Please reason step by step, and put your final answer within \\boxed{}."
        user = question

    text = f"<|im_start|>system\n{system}<|im_end|>\n"
    text += f"<|im_start|>user\n{user}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def build_prompt_math(question: str, use_sft_format: bool = False) -> str:
    """构建 MATH 的 chat template prompt"""
    if use_sft_format:
        system = ("Please reason step by step, and put your final answer within \\boxed{}. "
                  "All problems are verified and correct. "
                  "Do not question, doubt, or add notes about the problem statement. Focus only on solving.")
        text = f"<|im_start|>system\n{system}<|im_end|>\n"
        text += f"<|im_start|>user\n{question}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return text
    else:
        messages = build_messages(question, mode="cot")
        return messages_to_prompt(messages)


# ==================== 推理后端 ====================

def inference_vllm_offline(prompts: list, model_path: str, args) -> list:
    """使用 vLLM 离线推理（直接加载模型）"""
    from vllm import LLM, SamplingParams

    print(f"[INFO] 加载模型: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=0.0,       # greedy 解码，保证可复现
        max_tokens=args.max_tokens,
        top_p=1.0,
    )

    print(f"[INFO] 开始推理，共 {len(prompts)} 条...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        text = output.outputs[0].text
        results.append(text)
    return results


def inference_vllm_api(prompts: list, api_url: str, model_name: str, args) -> list:
    """通过 vLLM OpenAI-compatible API 推理"""
    from openai import OpenAI

    client = OpenAI(base_url=api_url, api_key="EMPTY")
    results = []

    print(f"[INFO] 通过 API 推理，共 {len(prompts)} 条...")
    for i, prompt in enumerate(prompts):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(prompts)}")
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            results.append(response.choices[0].text)
        except Exception as e:
            print(f"  [WARN] 第 {i} 条推理失败: {e}")
            results.append("")

    return results


# ==================== 评测主流程 ====================

def evaluate(dataset_name: str, data: list, model_outputs: list, extract_fn, detail_path: str = None):
    """计算评测指标，可选保存详细结果"""
    correct = 0
    total = len(data)
    wrong_examples = []
    detail_results = []

    for i, (item, output) in enumerate(zip(data, model_outputs)):
        pred = extract_fn(output)
        gold = item['gold_answer']
        is_correct = check_answer(pred, gold)

        if is_correct:
            correct += 1
        elif len(wrong_examples) < 5:  # 保存一些错误样例供分析
            wrong_examples.append({
                'index': i,
                'question': item['question'][:200],
                'gold': gold,
                'pred': pred,
                'model_output': output[:500],
            })

        # 保存每条的详细结果
        detail_results.append({
            'question': item['question'],
            'gold_answer': gold,
            'pred_answer': pred,
            'correct': is_correct,
            'pred_text': output,
        })

    acc = correct / total * 100

    print(f"\n{'='*60}")
    print(f"  {dataset_name} 评测结果")
    print(f"{'='*60}")
    print(f"  正确: {correct}/{total}")
    print(f"  准确率: {acc:.2f}%")

    # 如果是 MATH，按难度和类别统计
    if dataset_name == "MATH":
        print(f"\n  --- 按难度分 ---")
        level_stats = {}
        for item, output in zip(data, model_outputs):
            level = item.get('level', 'Unknown')
            pred = extract_fn(output)
            is_correct = check_answer(pred, item['gold_answer'])
            if level not in level_stats:
                level_stats[level] = {'correct': 0, 'total': 0}
            level_stats[level]['total'] += 1
            if is_correct:
                level_stats[level]['correct'] += 1

        for level in sorted(level_stats.keys()):
            s = level_stats[level]
            print(f"  {level}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")

        print(f"\n  --- 按类别分 ---")
        type_stats = {}
        for item, output in zip(data, model_outputs):
            t = item.get('type', 'Unknown')
            pred = extract_fn(output)
            is_correct = check_answer(pred, item['gold_answer'])
            if t not in type_stats:
                type_stats[t] = {'correct': 0, 'total': 0}
            type_stats[t]['total'] += 1
            if is_correct:
                type_stats[t]['correct'] += 1

        for t in sorted(type_stats.keys()):
            s = type_stats[t]
            print(f"  {t}: {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")

    # 打印错误样例
    if wrong_examples:
        print(f"\n  --- 部分错误样例 ---")
        for ex in wrong_examples[:3]:
            print(f"  [{ex['index']}] 标准答案: {ex['gold']}, 模型输出: {ex['pred']}")
            print(f"       题目: {ex['question'][:100]}...")
            print(f"       原始输出: {ex['model_output'][:200]}...")

    # 保存详细结果
    if detail_path:
        with open(detail_path, 'w', encoding='utf-8') as f:
            for r in detail_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\n  详细结果已保存: {detail_path}")

    return {
        'dataset': dataset_name,
        'accuracy': acc,
        'correct': correct,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser(description="数学能力评测脚本")
    parser.add_argument("--model", type=str, default=None, help="模型路径（离线推理）")
    parser.add_argument("--api_url", type=str, default=None, help="vLLM API 地址（如 http://localhost:8000/v1）")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Math-7B", help="API 模式下的模型名")
    parser.add_argument("--dataset", type=str, default="all", choices=["gsm8k", "math", "all"], help="评测数据集")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--gpu_util", type=float, default=0.85, help="GPU 显存利用率")
    parser.add_argument("--max_tokens", type=int, default=2048, help="最大生成 token 数")
    parser.add_argument("--max_model_len", type=int, default=4096, help="模型最大上下文长度")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径（JSON）")
    parser.add_argument("--limit", type=int, default=None, help="限制评测条数（调试用）")
    parser.add_argument("--sft_format", action="store_true",
                        help="使用 SFT 训练格式评测（system='You are Qwen...', user 加 instruction 前缀）")
    args = parser.parse_args()

    if args.model is None and args.api_url is None:
        parser.error("必须指定 --model（离线推理）或 --api_url（API 推理）之一")

    all_results = {}

    # ---- GSM8K ----
    if args.dataset in ["gsm8k", "all"]:
        print("\n" + "="*60)
        print("  加载 GSM8K 数据...")
        data = load_gsm8k()
        if args.limit:
            data = data[:args.limit]
        print(f"  共 {len(data)} 条")

        prompts = [build_prompt_gsm8k(item['question'], use_sft_format=args.sft_format) for item in data]

        if args.api_url:
            outputs = inference_vllm_api(prompts, args.api_url, args.model_name, args)
        else:
            outputs = inference_vllm_offline(prompts, args.model, args)

        detail_path = args.output.replace('.json', '_gsm8k_detail.jsonl') if args.output else None
        result = evaluate("GSM8K", data, outputs, extract_gsm8k_answer, detail_path=detail_path)
        all_results['gsm8k'] = result

    # ---- MATH ----
    if args.dataset in ["math", "all"]:
        print("\n" + "="*60)
        print("  加载 MATH 数据...")
        data = load_math()
        if args.limit:
            data = data[:args.limit]
        print(f"  共 {len(data)} 条")

        prompts = [build_prompt_math(item['question'], use_sft_format=args.sft_format) for item in data]

        if args.api_url:
            outputs = inference_vllm_api(prompts, args.api_url, args.model_name, args)
        else:
            outputs = inference_vllm_offline(prompts, args.model, args)

        detail_path = args.output.replace('.json', '_math_detail.jsonl') if args.output else None
        result = evaluate("MATH", data, outputs, extract_math_answer, detail_path=detail_path)
        all_results['math'] = result

    # ---- 保存结果 ----
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 结果已保存至: {args.output}")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  评测汇总")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {res['dataset']}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
