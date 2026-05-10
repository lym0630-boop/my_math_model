"""
DPO 数据构造 Pipeline（最终版）

新策略:
  1. 选一批 bootstrap 题（默认示例是 24K，也可扩到 60K）
  2. Student (Qwen2.5-Math-7B) 每题采样 8 次
  3. 分流:
     - 全对 (8/8) → 丢弃（太简单）
     - 答对 >= 半 (4~7/8) → on-policy pair（chosen=正确回答, rejected=错误回答）
     - 答对 < 半 (1~3/8) → 交给 Teacher（Student 不会做）
     - 全错 (0/8) → 交给 Teacher
  4. Teacher (DeepSeek-R1-32B) 对"不会做"的题采样 2 次
     - 2次都对 → Teacher 真会做，作为 chosen（配 Student 错误回答作 rejected）
     - 其他 → 丢弃（题太难，或 Teacher 也不稳定）
  5. 组装 ~8K DPO pairs

运行方式:
  # Step 1: Student 推理 (4×A100, ~3h)
  python3 dpo_final_pipeline.py --stage student

  # Step 2: Teacher 推理 (4×A100, ~1h)
  python3 dpo_final_pipeline.py --stage teacher

  # Step 3: 组装（无需GPU）
  python3 dpo_final_pipeline.py --stage assemble

  # 也支持显式指定一套新文件，避免覆盖旧版 v2 结果
  python3 dpo_final_pipeline.py \
    --stage student \
    --questions_file sft_data/dpo_questions_60k_bootstrap_v3.jsonl \
    --student_file sft_data/student_responses_60k_bootstrap_v3.jsonl
"""

# ===================== 兼容性修复 =====================
# transformers 5.x 移除了 all_special_tokens_extended 属性
# 但 vllm 0.11.0 仍在使用它，导致 AttributeError
# 这里做 monkey-patch: 用 all_special_tokens 替代
import transformers
if not hasattr(transformers.PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    @property
    def _all_special_tokens_extended(self):
        return self.all_special_tokens
    transformers.PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended
    print("[patch] 已修复 transformers 5.x 兼容性问题")
# ===================== END patch =====================

import json
import re
import random
import time
import argparse
import os
from collections import Counter
import statistics


BASE_DIR = '/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline'
SFT_DIR = os.path.join(BASE_DIR, 'sft_data')

DEFAULT_QUESTIONS_FILE = os.path.join(SFT_DIR, 'dpo_questions_24k_v2.jsonl')
DEFAULT_STUDENT_FILE = os.path.join(SFT_DIR, 'student_responses_24k_v2.jsonl')
DEFAULT_TEACHER_FILE = os.path.join(SFT_DIR, 'teacher_responses_24k_v2.jsonl')
DEFAULT_DPO_OUTPUT = os.path.join(SFT_DIR, 'dpo_train_final_v2.jsonl')

DEFAULT_STUDENT_MODEL = os.path.join(BASE_DIR, 'Qwen2.5-Math-7B-Instruct')
DEFAULT_TEACHER_MODEL = os.path.join(BASE_DIR, 'DeepSeek-R1-Distill-Qwen-32B')

THINK_SYSTEM_PROMPT = (
    "You must answer in this exact format:\n"
    "<think>\n"
    "your step-by-step reasoning\n"
    "</think>\n"
    "\\boxed{final answer}\n"
    "Always include both <think> and </think> tags, and always put the final answer within \\boxed{}."
)


def resolve_paths(args):
    return {
        'questions_file': args.questions_file,
        'student_file': args.student_file,
        'teacher_file': args.teacher_file,
        'dpo_output': args.dpo_output,
    }


# ===================== 答案提取与校验 =====================

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
    """LaTeX 表达式 → 数值"""
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
    # 纯数值
    cleaned = s.replace(',', '').replace(' ', '').replace('\\,', '')
    try:
        return float(cleaned)
    except ValueError:
        pass
    # a/b
    m = re.match(r'^(-?\d+\.?\d*)\s*/\s*(\d+\.?\d*)$', cleaned)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def check_answer(pred_val, gt_val, tol=1e-3):
    if pred_val is None or gt_val is None:
        return False
    if abs(gt_val) < 1e-10:
        return abs(pred_val) < tol
    if gt_val == int(gt_val) and abs(gt_val) < 1e9:
        return abs(pred_val - gt_val) < 0.5
    return (abs(pred_val - gt_val) / max(abs(gt_val), 1e-10) < tol
            or abs(pred_val - gt_val) < tol)


def extract_and_check(text, ground_truth):
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


def load_question_keys(questions_file):
    """加载当前题池的 question key，用于后续阶段只处理目标子集。"""
    question_keys = set()
    with open(questions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            question_keys.add(d['question'][:120].strip())
    return question_keys


def shard_output_path(path, shard_id, num_shards):
    """为分片输出生成独立文件名。"""
    if num_shards <= 1:
        return path
    if path.endswith('.jsonl'):
        return path[:-6] + f'.shard{shard_id}of{num_shards}.jsonl'
    return path + f'.shard{shard_id}of{num_shards}'


def get_shard_slice(total, num_shards, shard_id):
    """按连续区间切分样本，方便后续按 shard 顺序合并恢复原始顺序。"""
    if num_shards <= 1:
        return 0, total
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id={shard_id} 越界，num_shards={num_shards}")

    base = total // num_shards
    rem = total % num_shards
    start = shard_id * base + min(shard_id, rem)
    end = start + base + (1 if shard_id < rem else 0)
    return start, end


def merge_jsonl_shards(base_path, num_shards):
    """合并多个分片 jsonl 文件。"""
    merged = []
    for shard_id in range(num_shards):
        shard_path = shard_output_path(base_path, shard_id, num_shards)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"缺少分片文件: {shard_path}")
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    merged.append(line)

    with open(base_path, 'w') as f:
        for line in merged:
            f.write(line + '\n')

    print("已合并 %d 个分片到: %s" % (num_shards, base_path))
    print("合并后总行数: %d" % len(merged))


# ===================== Stage 1: Student 推理 =====================

def run_student(
    paths,
    student_model,
    tp_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    num_shards=1,
    shard_id=0,
):
    from vllm import LLM, SamplingParams

    # 加载题目
    data = []
    with open(paths['questions_file']) as f:
        for line in f:
            data.append(json.loads(line))
    total_questions = len(data)
    start_idx, end_idx = get_shard_slice(total_questions, num_shards, shard_id)
    data = data[start_idx:end_idx]
    output_path = shard_output_path(paths['student_file'], shard_id, num_shards)

    print("加载 %d 道题" % total_questions)
    if num_shards > 1:
        print("当前 Student 分片: %d/%d, 处理区间 [%d, %d), 共 %d 题" % (
            shard_id + 1, num_shards, start_idx, end_idx, len(data)))

    # 初始化模型
    print("加载 Student: %s" % student_model)
    llm = LLM(
        model=student_model,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        n=8,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 构造 prompt
    prompts = []
    for d in data:
        prompt = (
            "<|im_start|>system\n"
            + THINK_SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n" + d['question'] + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    # 分批推理
    batch_size = 2000
    all_outputs = []
    start = time.time()
    for bs in range(0, len(prompts), batch_size):
        be = min(bs + batch_size, len(prompts))
        print("  批次 %d/%d: 题目 %d~%d" % (
            bs // batch_size + 1,
            (len(prompts) + batch_size - 1) // batch_size,
            bs + 1, be))
        outputs = llm.generate(prompts[bs:be], sampling_params)
        all_outputs.extend(outputs)

    elapsed = time.time() - start
    print("Student 推理完成: %.0fs (%.1fmin)" % (elapsed, elapsed / 60))

    # 处理结果
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

        nc = sum(r['is_correct'] for r in responses)
        nt = len(responses)

        results.append({
            'question': d['question'],
            'ground_truth': gt,
            'num_correct': nc,
            'num_total': nt,
            'responses': responses,
        })

    # 写入
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 统计
    total = len(results)
    all_c = sum(1 for r in results if r['num_correct'] == r['num_total'])
    half_plus = sum(1 for r in results if r['num_total'] // 2 <= r['num_correct'] < r['num_total'])
    below_half = sum(1 for r in results if 0 < r['num_correct'] < r['num_total'] // 2)
    zero = sum(1 for r in results if r['num_correct'] == 0)

    print("\n" + "=" * 55)
    print("Student 采样统计")
    print("=" * 55)
    print("总题数: %d" % total)
    print("  全对 (8/8):      %5d (%.1f%%) → 丢弃" % (all_c, all_c / total * 100))
    print("  答对>=半 (4~7/8): %5d (%.1f%%) → on-policy pair" % (half_plus, half_plus / total * 100))
    print("  答对<半 (1~3/8):  %5d (%.1f%%) → 交给 Teacher" % (below_half, below_half / total * 100))
    print("  全错 (0/8):      %5d (%.1f%%) → 交给 Teacher" % (zero, zero / total * 100))
    print("\n需要 Teacher 推理: %d 题" % (below_half + zero))
    print("输出: %s" % output_path)


# ===================== Stage 2: Teacher 推理 =====================

def run_teacher(
    paths,
    teacher_model,
    tp_size=4,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    num_shards=1,
    shard_id=0,
):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # 加载需要 Teacher 推理的题（答对<半 + 全错）
    target_question_keys = load_question_keys(paths['questions_file'])
    need_teacher = []
    with open(paths['student_file']) as f:
        for line in f:
            d = json.loads(line)
            qkey = d['question'][:120].strip()
            if qkey not in target_question_keys:
                continue
            nc = d['num_correct']
            nt = d['num_total']
            if nc < nt // 2:  # 答对<半 和 全错 都交给 Teacher
                need_teacher.append(d)

    total_need_teacher = len(need_teacher)
    start_idx, end_idx = get_shard_slice(total_need_teacher, num_shards, shard_id)
    need_teacher = need_teacher[start_idx:end_idx]
    output_path = shard_output_path(paths['teacher_file'], shard_id, num_shards)

    print("需要 Teacher 推理: %d 题" % len(need_teacher))
    if num_shards > 1:
        print("当前 Teacher 分片: %d/%d, 原始待处理 %d 题, 当前区间 [%d, %d), 共 %d 题" % (
            shard_id + 1, num_shards, total_need_teacher, start_idx, end_idx, len(need_teacher)))

    if not need_teacher:
        print("没有需要 Teacher 推理的题，跳过")
        with open(output_path, 'w') as f:
            pass
        return

    # 初始化模型
    print("加载 Teacher: %s" % teacher_model)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    llm = LLM(
        model=teacher_model,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # 关键变化: n=2, temperature>0，采样两次
    sampling_params = SamplingParams(
        n=2,                    # 采样 2 次
        temperature=0.6,        # 需要一定随机性，两次独立采样
        top_p=0.9,
        max_tokens=4096,
        stop=["<｜end▁of▁sentence｜>"],
    )

    # 构造 prompt（格式指令 + 4-shot 示范）
    # 目的: 让 Teacher 的输出风格和 Student 尽量一致
    # 这样 DPO 的 chosen(Teacher) 和 rejected(Student) 风格统一
    # 模型只学推理质量差异，不会学到格式偏好
    #
    # 4-shot 覆盖: 代数、微积分、概率统计、组合概率
    # 每个示范都取自真实 Student 风格的数据

    FORMAT_INSTRUCTION = (
        "You must answer in this exact format: "
        "<think> your step-by-step reasoning </think> "
        "\\boxed{final answer}. "
        "Use concise derivations with key equations inside the think block. "
        "Always include both <think> and </think> tags, and always put the final answer within \\boxed{}."
    )

    FEWSHOT_EXAMPLES = [
        # Shot 1: 代数/递推
        {
            'q': (
                "Let $ f : \\mathbb{R} \\to \\mathbb{R} $ satisfy "
                "$ f(2x) - f(x) = (2x)^2 $ for all $ x $. "
                "Given $ f(1) = 1 $, determine $ f(8) $."
            ),
            'a': (
                "<think>\n"
                "The recurrence gives $ f(2x) = f(x) + 4x^2 $. Iteratively:\n"
                "$$\n"
                "f(2) = f(1) + 4(1)^2 = 1 + 4 = 5\n"
                "$$\n"
                "$$\n"
                "f(4) = f(2) + 4(2)^2 = 5 + 16 = 21\n"
                "$$\n"
                "$$\n"
                "f(8) = f(4) + 4(4)^2 = 21 + 64 = 85\n"
                "$$\n"
                "</think>\n\n"
                "$$\\boxed{85}$$"
            ),
        },
        # Shot 2: 微积分/定积分
        {
            'q': (
                "Compute the definite integral "
                "$\\int_1^3 (4t^3 - 2t)\\,dt$ given that "
                "$F(t) = t^4 - t^2$ is an antiderivative."
            ),
            'a': (
                "<think>\n"
                "By the Fundamental Theorem of Calculus:\n"
                "$$\n"
                "\\int_1^3 (4t^3 - 2t)\\,dt = F(3) - F(1)\n"
                "$$\n"
                "$$\n"
                "F(3) = 81 - 9 = 72,\\quad F(1) = 1 - 1 = 0\n"
                "$$\n"
                "$$\n"
                "72 - 0 = 72\n"
                "$$\n"
                "</think>\n\n"
                "$$\\boxed{72}$$"
            ),
        },
        # Shot 3: 概率统计
        {
            'q': (
                "In a survey of 80 students, 24 preferred chemistry. "
                "How many of 200 students in the school are expected "
                "to prefer chemistry?"
            ),
            'a': (
                "<think>\n"
                "The probability of preferring chemistry:\n"
                "$$\n"
                "\\frac{24}{80} = \\frac{3}{10}\n"
                "$$\n"
                "Applying to the school population:\n"
                "$$\n"
                "200 \\times \\frac{3}{10} = 60\n"
                "$$\n"
                "</think>\n\n"
                "$$\\boxed{60}$$"
            ),
        },
        # Shot 4: 组合/概率
        {
            'q': (
                "When two cards are drawn without replacement from a "
                "standard deck, what is the probability that both are red?"
            ),
            'a': (
                "<think>\n"
                "First card red: $ \\frac{26}{52} $. "
                "Given first is red, second red: $ \\frac{25}{51} $.\n"
                "$$\n"
                "P = \\frac{26}{52} \\times \\frac{25}{51} "
                "= \\frac{1}{2} \\times \\frac{25}{51} "
                "= \\frac{25}{102}\n"
                "$$\n"
                "</think>\n\n"
                "$$\\boxed{\\dfrac{25}{102}}$$"
            ),
        },
    ]

    prompts = []
    for d in need_teacher:
        messages = [{"role": "system", "content": FORMAT_INSTRUCTION}]
        for ex in FEWSHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex['q']})
            messages.append({"role": "assistant", "content": ex['a']})
        messages.append({"role": "user", "content": d['question']})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # 分批推理
    batch_size = 500
    all_outputs = []
    start = time.time()
    for bs in range(0, len(prompts), batch_size):
        be = min(bs + batch_size, len(prompts))
        print("  批次 %d/%d: 题目 %d~%d" % (
            bs // batch_size + 1,
            (len(prompts) + batch_size - 1) // batch_size,
            bs + 1, be))
        outputs = llm.generate(prompts[bs:be], sampling_params)
        all_outputs.extend(outputs)

    elapsed = time.time() - start
    print("Teacher 推理完成: %.0fs (%.1fmin)" % (elapsed, elapsed / 60))

    # 处理结果: 两次都要答对才算 Teacher 会做
    results = []
    both_correct = 0
    one_correct = 0
    both_wrong = 0

    for d, output in zip(need_teacher, all_outputs):
        gt = d['ground_truth']

        resp1_text = ensure_think_block(output.outputs[0].text.strip())
        resp2_text = ensure_think_block(output.outputs[1].text.strip())

        pred1, correct1 = extract_and_check(resp1_text, gt)
        pred2, correct2 = extract_and_check(resp2_text, gt)

        if correct1 and correct2:
            both_correct += 1
            status = 'both_correct'
        elif correct1 or correct2:
            one_correct += 1
            status = 'one_correct'
        else:
            both_wrong += 1
            status = 'both_wrong'

        # 如果两次都对，用第一次的回答作为 chosen，并保留 think tag
        results.append({
            'question': d['question'],
            'ground_truth': gt,
            'status': status,
            'teacher_response': resp1_text if status == 'both_correct' else None,
            'resp1_correct': correct1,
            'resp2_correct': correct2,
            # 保留 Student 的错误回答用于配对
            'student_wrong_responses': [r['text'] for r in d['responses'] if not r['is_correct']],
        })

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print("\n" + "=" * 55)
    print("Teacher 采样统计（每题2次）")
    print("=" * 55)
    total = len(results)
    print("总题数: %d" % total)
    print("  两次全对: %5d (%.1f%%) → ✅ 作为 chosen" % (both_correct, both_correct / total * 100))
    print("  只对一次: %5d (%.1f%%) → ❌ 丢弃（不稳定）" % (one_correct, one_correct / total * 100))
    print("  两次全错: %5d (%.1f%%) → ❌ 丢弃（题太难）" % (both_wrong, both_wrong / total * 100))
    print("输出: %s" % output_path)


# ===================== Stage 3: 组装 DPO pairs =====================

SYSTEM_PROMPT = THINK_SYSTEM_PROMPT

def build_dpo_prompt(question):
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def length_ratio_ok(chosen, rejected, max_ratio=2.5):
    """
    长度过滤: 防止 DPO 学到 "短=好" 的虚假信号

    问题: rejected（答错的回答）往往比 chosen 长很多，因为模型答不出来时
    会反复尝试、绕圈子。如果不过滤，DPO 会学到 "回答越短越好"。

    策略: 过滤掉 chosen 和 rejected 长度比超过 max_ratio 的 pair
    """
    cl = len(chosen)
    rl = len(rejected)
    if cl == 0 or rl == 0:
        return False
    ratio = max(cl, rl) / min(cl, rl)
    return ratio <= max_ratio


def best_length_match(chosen_list, rejected_list, max_pairs=2, max_ratio=2.5):
    """
    从 chosen/rejected 列表中选择长度最匹配的 pair
    避免系统性的 chosen 比 rejected 短的偏差
    """
    if not chosen_list or not rejected_list:
        return []

    pairs = []
    used_c = set()
    used_r = set()

    # 按长度差排序所有可能的配对
    candidates = []
    for ci, c in enumerate(chosen_list):
        for ri, r in enumerate(rejected_list):
            diff = abs(len(c['text']) - len(r['text']))
            ratio = max(len(c['text']), len(r['text'])) / max(min(len(c['text']), len(r['text'])), 1)
            if ratio <= max_ratio:
                candidates.append((diff, ci, ri))

    candidates.sort()  # 长度差最小的优先

    for diff, ci, ri in candidates:
        if len(pairs) >= max_pairs:
            break
        if ci in used_c or ri in used_r:
            continue
        pairs.append((chosen_list[ci]['text'], rejected_list[ri]['text']))
        used_c.add(ci)
        used_r.add(ri)

    return pairs


def run_assemble(paths, target=8000, seed=42):
    random.seed(seed)

    len_filtered = 0  # 被长度过滤掉的计数
    target_question_keys = load_question_keys(paths['questions_file'])

    # ---- on-policy pairs: 答对>=半的题 ----
    onpolicy_pairs = []
    with open(paths['student_file']) as f:
        for line in f:
            d = json.loads(line)
            qkey = d['question'][:120].strip()
            if qkey not in target_question_keys:
                continue
            nc = d['num_correct']
            nt = d['num_total']

            # 答对 >= 半 且 不是全对
            if nc >= nt // 2 and nc < nt:
                correct_resps = [r for r in d['responses'] if r['is_correct']]
                wrong_resps = [r for r in d['responses'] if not r['is_correct']]

                # 用长度匹配选最佳配对（最多2对）
                matched = best_length_match(correct_resps, wrong_resps, max_pairs=2)
                for chosen, rejected in matched:
                    len_filtered_matched = 0
                    onpolicy_pairs.append({
                        'prompt': build_dpo_prompt(d['question']),
                        'chosen': chosen,
                        'rejected': rejected,
                        'source': 'on_policy',
                        'ground_truth': d['ground_truth'],
                    })

    # ---- teacher-chosen pairs: Teacher 两次全对 ----
    teacher_pairs = []
    if os.path.exists(paths['teacher_file']):
        with open(paths['teacher_file']) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                qkey = t['question'][:120].strip()
                if qkey not in target_question_keys:
                    continue
                if t['status'] != 'both_correct':
                    continue
                if not t['student_wrong_responses']:
                    continue

                wrong_list = t['student_wrong_responses']
                random.shuffle(wrong_list)

                chosen = t['teacher_response']
                rejected = wrong_list[0]
                # 长度过滤
                if not length_ratio_ok(chosen, rejected):
                    len_filtered += 1
                    continue

                teacher_pairs.append({
                    'prompt': build_dpo_prompt(t['question']),
                    'chosen': chosen,
                    'rejected': rejected,
                    'source': 'teacher_chosen',
                    'ground_truth': t['ground_truth'],
                })

    print("on-policy pairs: %d" % len(onpolicy_pairs))
    print("teacher pairs:   %d" % len(teacher_pairs))
    print("长度过滤掉:      %d (chosen/rejected 长度比 > 2.5x)" % len_filtered)

    # ---- 混合 ----
    total = len(onpolicy_pairs) + len(teacher_pairs)
    if total <= target:
        final = onpolicy_pairs + teacher_pairs
    else:
        # on-policy 优先，teacher 补足
        if len(onpolicy_pairs) >= target:
            random.shuffle(onpolicy_pairs)
            final = onpolicy_pairs[:target]
        else:
            need = target - len(onpolicy_pairs)
            random.shuffle(teacher_pairs)
            final = onpolicy_pairs + teacher_pairs[:need]

    random.shuffle(final)

    # 写入
    with open(paths['dpo_output'], 'w') as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # ---- 统计报告 ----
    source_dist = Counter(p['source'] for p in final)
    chosen_lens = [len(p['chosen']) for p in final]
    rejected_lens = [len(p['rejected']) for p in final]

    print("\n" + "=" * 60)
    print("最终 DPO 训练数据")
    print("=" * 60)
    print("总 pairs: %d / 目标 %d  %s" % (
        len(final), target, "✅" if len(final) >= target * 0.9 else "⚠️ 不足"))

    print("\n来源:")
    for src, cnt in source_dist.most_common():
        print("  %s: %d (%.1f%%)" % (src, cnt, cnt / len(final) * 100))

    if chosen_lens:
        print("\nchosen 长度:  median=%d, mean=%d" % (
            statistics.median(chosen_lens), statistics.mean(chosen_lens)))
        print("rejected 长度: median=%d, mean=%d" % (
            statistics.median(rejected_lens), statistics.mean(rejected_lens)))

        # 长度偏差分析：检测系统性 chosen 比 rejected 短的情况
        len_diffs = [len(p['chosen']) - len(p['rejected']) for p in final]
        shorter_cnt = sum(1 for d in len_diffs if d < 0)
        longer_cnt = sum(1 for d in len_diffs if d > 0)
        print("\n长度偏差分析:")
        print("  chosen 比 rejected 短: %d (%.1f%%)" % (shorter_cnt, shorter_cnt / len(final) * 100))
        print("  chosen 比 rejected 长: %d (%.1f%%)" % (longer_cnt, longer_cnt / len(final) * 100))
        print("  长度差 median: %d chars" % statistics.median(len_diffs))
        if abs(statistics.median(len_diffs)) > 100:
            print("  ⚠️ 警告: 存在系统性长度偏差，DPO 可能学到长度偏好")

    print("\n输出: %s" % paths['dpo_output'])
    print("文件大小: %.1fMB" % (os.path.getsize(paths['dpo_output']) / 1024 / 1024))


# ===================== 主入口 =====================

def main():
    parser = argparse.ArgumentParser(description='DPO 数据构造 Pipeline（最终版）')
    parser.add_argument('--stage', required=True,
                        choices=['student', 'teacher', 'assemble', 'merge_student', 'merge_teacher'],
                        help='student=Student推理, teacher=Teacher推理, assemble=组装, merge_student/merge_teacher=合并分片')
    parser.add_argument('--tp_size', type=int, default=4)
    parser.add_argument('--target', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--questions_file', default=DEFAULT_QUESTIONS_FILE,
                        help='待采样题池路径')
    parser.add_argument('--student_file', default=DEFAULT_STUDENT_FILE,
                        help='Student 多采样结果路径')
    parser.add_argument('--teacher_file', default=DEFAULT_TEACHER_FILE,
                        help='Teacher rescue 结果路径')
    parser.add_argument('--dpo_output', default=DEFAULT_DPO_OUTPUT,
                        help='最终 DPO pair 输出路径')
    parser.add_argument('--student_model', default=DEFAULT_STUDENT_MODEL,
                        help='Student 模型路径')
    parser.add_argument('--teacher_model', default=DEFAULT_TEACHER_MODEL,
                        help='Teacher 模型路径')
    parser.add_argument('--student_max_model_len', type=int, default=4096,
                        help='Student vLLM max_model_len')
    parser.add_argument('--teacher_max_model_len', type=int, default=8192,
                        help='Teacher vLLM max_model_len')
    parser.add_argument('--student_gpu_memory_utilization', type=float, default=0.90,
                        help='Student vLLM gpu_memory_utilization')
    parser.add_argument('--teacher_gpu_memory_utilization', type=float, default=0.90,
                        help='Teacher vLLM gpu_memory_utilization')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='把数据切成多少片并行跑；例如 8 卡可用 2 片 × 4 卡')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='当前进程负责的分片编号，从 0 开始')
    args = parser.parse_args()
    paths = resolve_paths(args)

    if args.stage == 'student':
        print("=" * 60)
        print("Stage 1: Student 推理")
        print("=" * 60)
        run_student(
            paths,
            args.student_model,
            args.tp_size,
            args.student_max_model_len,
            args.student_gpu_memory_utilization,
            args.num_shards,
            args.shard_id,
        )

    elif args.stage == 'teacher':
        print("=" * 60)
        print("Stage 2: Teacher 推理 (2次采样，两次全对才算会)")
        print("=" * 60)
        run_teacher(
            paths,
            args.teacher_model,
            args.tp_size,
            args.teacher_max_model_len,
            args.teacher_gpu_memory_utilization,
            args.num_shards,
            args.shard_id,
        )

    elif args.stage == 'assemble':
        print("=" * 60)
        print("Stage 3: 组装 DPO pairs")
        print("=" * 60)
        run_assemble(paths, args.target, args.seed)

    elif args.stage == 'merge_student':
        merge_jsonl_shards(paths['student_file'], args.num_shards)

    elif args.stage == 'merge_teacher':
        merge_jsonl_shards(paths['teacher_file'], args.num_shards)


if __name__ == '__main__':
    main()
