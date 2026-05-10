"""
前缀引导 Warm-Start Pipeline

在 GRPO 之前，对 DPO-v2 模型做定向抢救：
  - 找出 Student 8次全错的 hard bucket 题目
  - 截取参考解的前缀（短/中/长三档），让模型续写
  - 过滤出答案正确且非抄袭的样本
  - 组装为小规模 SFT 数据，做 warm-start

运行方式:
  python3 prefix_guided_warmstart.py --stage prepare      # 准备候选（无需GPU）
  python3 prefix_guided_warmstart.py --stage generate      # 模型续写（需要GPU）
  python3 prefix_guided_warmstart.py --stage assemble      # 过滤组装（无需GPU）
  python3 prefix_guided_warmstart.py --stage evaluate      # 评测 gate（需要GPU）
"""

# ===================== 兼容性修复（延迟到需要 vLLM 时再执行）=====================
def _patch_transformers():
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
from collections import Counter, defaultdict


BASE_DIR = '/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline'
SFT_DIR = os.path.join(BASE_DIR, 'sft_data')

STUDENT_FILE = os.path.join(SFT_DIR, 'student_responses_24k_v2.jsonl')
TEACHER_FILE = os.path.join(SFT_DIR, 'teacher_responses_24k_v2.jsonl')
QUESTIONS_FILE = os.path.join(SFT_DIR, 'dpo_questions_24k_v2.jsonl')

CANDIDATES_FILE = os.path.join(SFT_DIR, 'warmstart_candidates.jsonl')
GENERATIONS_FILE = os.path.join(SFT_DIR, 'warmstart_generations.jsonl')
SFT_OUTPUT = os.path.join(SFT_DIR, 'warmstart_sft_train.json')
EVAL_FILE = os.path.join(SFT_DIR, 'warmstart_eval_500.jsonl')

MODEL_PATH = os.path.join(BASE_DIR, 'Qwen2.5-Math-7B-DPO-v2')

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# ===================== 答案提取与校验（复用 dpo_final_pipeline.py）=====================

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


# ===================== 前缀截取 =====================

# 语义断点模式：优先在这些位置切
BREAKPOINT_PATTERNS = [
    r'\$\$\s*\n',          # $$ 后跟换行（LaTeX display math 结束）
    r'\n\s*\n',             # 空行（段落分隔）
    r'\n\s*[-\*]\s',        # 无序列表项开头
    r'\n\s*\d+[\.\)]\s',   # 有序列表项开头（1. 或 1) ）
    r'\n\s*\*\*',           # 加粗标题开头
]


def find_semantic_breakpoints(text):
    """找出文本中所有语义断点的位置（返回断点结束位置的列表）"""
    breakpoints = set()
    for pat in BREAKPOINT_PATTERNS:
        for m in re.finditer(pat, text):
            breakpoints.add(m.end())

    # 补充：所有 \n 位置作为兜底断点
    for m in re.finditer(r'\n', text):
        breakpoints.add(m.end())

    return sorted(breakpoints)


def cut_prefix(text, target_ratio, breakpoints):
    """在目标比例附近找最近的语义断点截取前缀

    Args:
        text: 完整参考解
        target_ratio: 目标截取比例 (0.0-1.0)
        breakpoints: 排序后的断点位置列表

    Returns:
        prefix: 截取的前缀文本，如果找不到合适断点返回 None
    """
    target_pos = int(len(text) * target_ratio)

    if not breakpoints:
        return None

    # 找离 target_pos 最近的断点
    best = min(breakpoints, key=lambda bp: abs(bp - target_pos))

    # 断点不能太靠头（< 10%）或太靠尾（> 80%）
    if best < len(text) * 0.05 or best > len(text) * 0.80:
        # 尝试找次优的
        candidates = [bp for bp in breakpoints if len(text) * 0.05 <= bp <= len(text) * 0.80]
        if candidates:
            best = min(candidates, key=lambda bp: abs(bp - target_pos))
        else:
            return None

    return text[:best]


def check_answer_leakage(prefix, ground_truth):
    """检查前缀是否泄露了最终答案

    检查 \\boxed{ground_truth} 或直接包含 GT 数值
    """
    gt_str = ground_truth.strip()

    # 检查 \boxed{answer}
    if f'\\boxed{{{gt_str}}}' in prefix:
        return True

    # 检查纯数值是否出现在最后 20% 的文本中
    # （GT 可能在中间推理步骤中出现，只检查尾部以减少误报）
    tail = prefix[int(len(prefix) * 0.8):]
    # 对整数 GT，检查是否出现在运算结果的位置
    if re.match(r'^-?\d+\.?\d*$', gt_str):
        # 检查 = GT 或 GT\n 这种模式
        if re.search(r'=\s*' + re.escape(gt_str) + r'\s*[\n$\\,.]', tail):
            return True

    return False


def truncate_before_leakage(prefix, ground_truth, breakpoints_in_prefix):
    """如果前缀泄露了答案，截短到泄露位置之前"""
    gt_str = ground_truth.strip()

    # 找 \boxed{GT} 位置
    boxed_pat = f'\\boxed{{{gt_str}}}'
    idx = prefix.find(boxed_pat)
    if idx == -1:
        # 找 = GT 模式
        m = re.search(r'=\s*' + re.escape(gt_str) + r'\s*[\n$\\,.]', prefix)
        if m and m.start() > len(prefix) * 0.8:
            idx = m.start()

    if idx == -1:
        return prefix  # 没有泄露

    # 截到泄露位置之前最近的断点
    candidates = [bp for bp in breakpoints_in_prefix if bp < idx]
    if candidates:
        return prefix[:candidates[-1]]
    else:
        return prefix[:max(idx - 50, int(len(prefix) * 0.3))]


# ===================== Stage 1: 准备候选数据 =====================

def run_prepare(seed=42, max_candidates=4000):
    random.seed(seed)

    print("=" * 60)
    print("Stage 1: 准备前缀引导候选数据")
    print("=" * 60)

    # 1. 加载参考解（从 dpo_questions）
    print("[1/5] 加载参考解...")
    ref_answers = {}
    with open(QUESTIONS_FILE) as f:
        for line in f:
            d = json.loads(line)
            ref_answers[d['question'][:120]] = d['answer']
    print("  参考解数量: %d" % len(ref_answers))

    # 2. 加载 Teacher 状态
    print("[2/5] 加载 Teacher 状态...")
    teacher_status = {}
    with open(TEACHER_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            teacher_status[d['question'][:120]] = d['status']
    print("  Teacher 结果: %d" % len(teacher_status))

    # 3. 筛选 Student 全错题
    print("[3/5] 筛选 Student 全错题...")
    hard_bucket = []
    with open(STUDENT_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d['num_correct'] != 0:
                continue
            qk = d['question'][:120]
            if qk not in ref_answers:
                continue
            hard_bucket.append({
                'question': d['question'],
                'ground_truth': d['ground_truth'],
                'answer': ref_answers[qk],
                'teacher_status': teacher_status.get(qk, 'unknown'),
            })
    print("  全错题: %d (全部有参考解)" % len(hard_bucket))

    # 4. 按优先级排序：both_correct > one_correct > both_wrong
    priority = {'both_correct': 0, 'one_correct': 1, 'both_wrong': 2, 'unknown': 3}
    hard_bucket.sort(key=lambda x: priority.get(x['teacher_status'], 3))

    # 统计
    ts_counts = Counter(d['teacher_status'] for d in hard_bucket)
    print("  Teacher 状态分布:")
    for s, c in ts_counts.most_common():
        print("    %s: %d" % (s, c))

    # 截取前 max_candidates 题
    if len(hard_bucket) > max_candidates:
        hard_bucket = hard_bucket[:max_candidates]
        print("  截取前 %d 题（优先 Teacher 会的题）" % max_candidates)

    # 5. 对每题生成三档前缀
    print("[4/5] 生成三档前缀...")
    # 前缀比例：短20%, 中40%, 长60%
    PREFIX_RATIOS = {
        'short': 0.20,
        'mid': 0.40,
        'long': 0.60,
    }

    candidates = []
    prefix_stats = Counter()
    leakage_fixes = 0

    for d in hard_bucket:
        answer_text = d['answer']
        breakpoints = find_semantic_breakpoints(answer_text)

        prefixes = {}
        valid = True

        for ptype, ratio in PREFIX_RATIOS.items():
            prefix = cut_prefix(answer_text, ratio, breakpoints)
            if prefix is None or len(prefix) < 20:
                prefix_stats['skip_no_breakpoint'] += 1
                valid = False
                break

            # 答案泄露检查
            if check_answer_leakage(prefix, d['ground_truth']):
                bp_in_prefix = [bp for bp in breakpoints if bp <= len(prefix)]
                prefix = truncate_before_leakage(prefix, d['ground_truth'], bp_in_prefix)
                leakage_fixes += 1
                if len(prefix) < 20:
                    valid = False
                    break

            prefixes[ptype] = prefix

        if not valid:
            continue

        candidates.append({
            'question': d['question'],
            'ground_truth': d['ground_truth'],
            'answer': answer_text,
            'teacher_status': d['teacher_status'],
            'prefix_short': prefixes['short'],
            'prefix_mid': prefixes['mid'],
            'prefix_long': prefixes['long'],
        })

    # 写入
    print("[5/5] 写入 %s..." % CANDIDATES_FILE)
    with open(CANDIDATES_FILE, 'w') as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    # 统计
    print("\n" + "=" * 60)
    print("候选统计")
    print("=" * 60)
    print("总候选题: %d" % len(candidates))
    print("答案泄露修复: %d 次" % leakage_fixes)

    ts_final = Counter(c['teacher_status'] for c in candidates)
    print("Teacher 状态:")
    for s, c in ts_final.most_common():
        print("  %s: %d" % (s, c))

    # 前缀长度分布
    import numpy as np
    for ptype in ['short', 'mid', 'long']:
        lens = [len(c['prefix_' + ptype]) for c in candidates]
        arr = np.array(lens)
        print("%s 前缀长度: median=%d, mean=%d" % (ptype, np.median(arr), np.mean(arr)))

    print("\n输出: %s" % CANDIDATES_FILE)


# ===================== Stage 2: 模型续写 =====================

def run_generate(tp_size=4):
    _patch_transformers()
    from vllm import LLM, SamplingParams

    # 加载候选
    candidates = []
    with open(CANDIDATES_FILE) as f:
        for line in f:
            candidates.append(json.loads(line))
    print("加载 %d 道候选题" % len(candidates))

    # 初始化模型
    print("加载模型: %s" % MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        n=4,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 构造所有 prompt（每题3个前缀 × 1组）
    all_prompts = []
    all_meta = []  # (题目索引, 前缀类型)

    for idx, c in enumerate(candidates):
        for ptype in ['short', 'mid', 'long']:
            prefix = c['prefix_' + ptype]
            prompt = (
                "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
                "<|im_start|>user\n" + c['question'] + "<|im_end|>\n"
                "<|im_start|>assistant\n" + prefix
            )
            all_prompts.append(prompt)
            all_meta.append((idx, ptype))

    print("总 prompt 数: %d (每题3个前缀)" % len(all_prompts))

    # 分批推理
    batch_size = 1000
    all_outputs = []
    start = time.time()
    for bs in range(0, len(all_prompts), batch_size):
        be = min(bs + batch_size, len(all_prompts))
        print("  批次 %d/%d: prompt %d~%d" % (
            bs // batch_size + 1,
            (len(all_prompts) + batch_size - 1) // batch_size,
            bs + 1, be))
        outputs = llm.generate(all_prompts[bs:be], sampling_params)
        all_outputs.extend(outputs)

    elapsed = time.time() - start
    print("推理完成: %.0fs (%.1fmin)" % (elapsed, elapsed / 60))

    # 处理结果
    results = []
    total_correct = 0
    total_generated = 0

    for (idx, ptype), output in zip(all_meta, all_outputs):
        c = candidates[idx]
        prefix = c['prefix_' + ptype]

        for comp in output.outputs:
            continuation = comp.text.strip()
            full_response = prefix + continuation
            pred, is_correct = extract_and_check(full_response, c['ground_truth'])

            total_generated += 1
            if is_correct:
                total_correct += 1

            results.append({
                'question': c['question'],
                'ground_truth': c['ground_truth'],
                'answer': c['answer'],  # 完整参考解（用于后续过滤）
                'teacher_status': c['teacher_status'],
                'prefix_type': ptype,
                'prefix': prefix,
                'continuation': continuation,
                'full_response': full_response,
                'predicted_answer': pred,
                'is_correct': is_correct,
            })

    # 写入
    with open(GENERATIONS_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 统计
    print("\n" + "=" * 60)
    print("续写统计")
    print("=" * 60)
    print("总生成: %d" % total_generated)
    print("答案正确: %d (%.1f%%)" % (total_correct, total_correct / max(total_generated, 1) * 100))

    # 按前缀类型统计
    for ptype in ['short', 'mid', 'long']:
        pt_total = sum(1 for r in results if r['prefix_type'] == ptype)
        pt_correct = sum(1 for r in results if r['prefix_type'] == ptype and r['is_correct'])
        print("  %s: %d/%d (%.1f%%)" % (ptype, pt_correct, pt_total, pt_correct / max(pt_total, 1) * 100))

    # 有至少1次正确的题数
    correct_questions = set()
    for r in results:
        if r['is_correct']:
            correct_questions.add(r['question'][:120])
    print("有正确答案的题: %d / %d (%.1f%%)" % (
        len(correct_questions), len(candidates),
        len(correct_questions) / max(len(candidates), 1) * 100))

    print("\n输出: %s" % GENERATIONS_FILE)


# ===================== 过滤工具 =====================

def longest_common_substring(s1, s2):
    """计算两个字符串的最长公共子串长度"""
    if not s1 or not s2:
        return 0
    # 优化：限制长度避免 O(n*m) 太慢
    s1 = s1[:3000]
    s2 = s2[:3000]

    m, n = len(s1), len(s2)
    max_len = 0
    # 滚动数组优化空间
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
                max_len = max(max_len, curr[j])
        prev = curr
    return max_len


def rouge_l_score(hypothesis, reference):
    """计算 ROUGE-L F1 分数（基于最长公共子序列）"""
    if not hypothesis or not reference:
        return 0.0

    # 用词级别的 LCS
    hyp_words = hypothesis.split()
    ref_words = reference.split()

    if not hyp_words or not ref_words:
        return 0.0

    # 限制长度
    hyp_words = hyp_words[:500]
    ref_words = ref_words[:500]

    m, n = len(hyp_words), len(ref_words)
    # DP 求 LCS 长度
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if hyp_words[i-1] == ref_words[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr

    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ===================== Stage 3: 过滤与组装 =====================

def run_assemble(seed=42, eval_holdout=500, max_per_question=2):
    random.seed(seed)

    print("=" * 60)
    print("Stage 3: 过滤与组装 warm-start SFT 数据")
    print("=" * 60)

    # 加载所有生成结果
    all_gens = []
    with open(GENERATIONS_FILE) as f:
        for line in f:
            all_gens.append(json.loads(line))
    print("加载 %d 条生成结果" % len(all_gens))

    # 过滤
    filter_stats = Counter()
    passed = []

    for g in all_gens:
        filter_stats['总数'] += 1

        # 1. 答案必须正确
        if not g['is_correct']:
            filter_stats['过滤: 答案错误'] += 1
            continue

        continuation = g['continuation']
        prefix = g['prefix']
        reference_remainder = g['answer'][len(prefix):]  # 参考解去掉前缀后的部分

        # 2. 续写最小长度
        if len(continuation) < 100:
            filter_stats['过滤: 续写太短(<100)'] += 1
            continue

        # 3. 答案泄露检查：续写开头不能直接出现 GT
        gt_str = g['ground_truth'].strip()
        if gt_str in continuation[:50]:
            filter_stats['过滤: 答案泄露'] += 1
            continue

        # 4. 连续抄写检测（LCS）
        lcs_len = longest_common_substring(continuation, reference_remainder)
        if lcs_len > 50:
            filter_stats['过滤: 连续抄写(LCS>50)'] += 1
            continue

        # 5. ROUGE-L 检查
        rouge = rouge_l_score(continuation, reference_remainder)
        if rouge > 0.7:
            filter_stats['过滤: ROUGE-L>0.7'] += 1
            continue

        filter_stats['通过'] += 1
        passed.append({
            **g,
            'lcs_len': lcs_len,
            'rouge_l': rouge,
        })

    # 打印过滤统计
    print("\n过滤统计:")
    for key in ['总数', '过滤: 答案错误', '过滤: 续写太短(<100)',
                 '过滤: 答案泄露', '过滤: 连续抄写(LCS>50)',
                 '过滤: ROUGE-L>0.7', '通过']:
        if key in filter_stats:
            print("  %s: %d" % (key, filter_stats[key]))

    # 按题目分组，每题选最佳 1-2 条
    by_question = defaultdict(list)
    for p in passed:
        by_question[p['question'][:120]].append(p)

    print("\n有通过样本的题数: %d" % len(by_question))

    # 选择逻辑：
    # - 优先短前缀（模型自己推理更多）
    # - 同前缀类型下选 ROUGE 最低的（最不像抄的）
    prefix_priority = {'short': 0, 'mid': 1, 'long': 2}
    selected = []

    for qk, items in by_question.items():
        # 按：前缀类型优先级 → ROUGE-L 升序 排序
        items.sort(key=lambda x: (prefix_priority.get(x['prefix_type'], 9), x['rouge_l']))

        # 取最好的 1 条
        selected.append(items[0])

        # 如果有不同前缀类型的第二条，且和第一条 ROUGE 互相 < 0.5，可以取第 2 条
        if len(items) >= 2 and max_per_question >= 2:
            second = items[1]
            if second['prefix_type'] != items[0]['prefix_type']:
                inter_rouge = rouge_l_score(items[0]['continuation'], second['continuation'])
                if inter_rouge < 0.5:
                    selected.append(second)

    print("最终选取: %d 条 (来自 %d 题)" % (len(selected), len(by_question)))

    # 分出评测 hold-out（从未被选中的全错题中选）
    selected_qks = set(s['question'][:120] for s in selected)

    # 加载所有全错题用于 hold-out
    all_hard = []
    with open(STUDENT_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d['num_correct'] == 0:
                qk = d['question'][:120]
                if qk not in selected_qks:
                    all_hard.append({
                        'question': d['question'],
                        'ground_truth': d['ground_truth'],
                    })

    random.shuffle(all_hard)
    eval_set = all_hard[:eval_holdout]
    print("评测集: %d 道题 (从非训练全错题中选)" % len(eval_set))

    # 写评测集
    with open(EVAL_FILE, 'w') as f:
        for e in eval_set:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    # 组装 SFT 数据（LLaMA-Factory sharegpt 格式）
    # 前缀直接拼在 assistant 开头：response = prefix + continuation
    sft_data = []
    for s in selected:
        sft_data.append({
            "conversations": [
                {"from": "human", "value": s['question']},
                {"from": "gpt", "value": s['full_response']},
            ],
            "system": SYSTEM_PROMPT,
        })

    # 打乱
    random.shuffle(sft_data)

    with open(SFT_OUTPUT, 'w') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # 统计报告
    import statistics
    cont_lens = [len(s['continuation']) for s in selected]
    full_lens = [len(s['full_response']) for s in selected]
    rouge_scores = [s['rouge_l'] for s in selected]

    prefix_dist = Counter(s['prefix_type'] for s in selected)
    teacher_dist = Counter(s['teacher_status'] for s in selected)

    print("\n" + "=" * 60)
    print("SFT 数据统计")
    print("=" * 60)
    print("总条数: %d" % len(sft_data))
    print("\n前缀类型分布:")
    for pt, cnt in prefix_dist.most_common():
        print("  %s: %d (%.1f%%)" % (pt, cnt, cnt / len(sft_data) * 100))
    print("\nTeacher 状态分布:")
    for ts, cnt in teacher_dist.most_common():
        print("  %s: %d (%.1f%%)" % (ts, cnt, cnt / len(sft_data) * 100))
    print("\n续写长度: median=%d, mean=%d" % (
        statistics.median(cont_lens), statistics.mean(cont_lens)))
    print("完整回答长度: median=%d, mean=%d" % (
        statistics.median(full_lens), statistics.mean(full_lens)))
    print("ROUGE-L: median=%.3f, mean=%.3f, max=%.3f" % (
        statistics.median(rouge_scores), statistics.mean(rouge_scores), max(rouge_scores)))
    print("\n输出: %s" % SFT_OUTPUT)
    print("评测集: %s" % EVAL_FILE)
    print("文件大小: %.1fMB" % (os.path.getsize(SFT_OUTPUT) / 1024 / 1024))


# ===================== Stage 4: 评测 gate =====================

def run_evaluate(model_path, tp_size=4):
    _patch_transformers()
    from vllm import LLM, SamplingParams

    # 加载评测集
    eval_data = []
    with open(EVAL_FILE) as f:
        for line in f:
            eval_data.append(json.loads(line))
    print("评测集: %d 道题" % len(eval_data))

    # 初始化模型
    print("加载模型: %s" % model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        n=8,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 构造 prompt（无前缀，纯题目）
    prompts = []
    for d in eval_data:
        prompt = (
            "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n" + d['question'] + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    # 推理
    print("开始推理 (%d 题 × 8 次 = %d 次生成)..." % (len(prompts), len(prompts) * 8))
    start = time.time()
    all_outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    print("推理完成: %.0fs" % elapsed)

    # 统计
    solved = 0
    total = len(eval_data)

    for d, output in zip(eval_data, all_outputs):
        gt = d['ground_truth']
        any_correct = False
        for comp in output.outputs:
            _, is_correct = extract_and_check(comp.text.strip(), gt)
            if is_correct:
                any_correct = True
                break
        if any_correct:
            solved += 1

    solve_rate = solved / max(total, 1) * 100

    print("\n" + "=" * 60)
    print("评测 Gate 结果")
    print("=" * 60)
    print("模型: %s" % model_path)
    print("评测题数: %d (全部是 DPO-v2 模型 8 次全错的难题)" % total)
    print("解决题数: %d" % solved)
    print("Solve Rate: %.1f%%" % solve_rate)
    print()
    if solve_rate >= 5.0:
        print("✅ 通过 gate (≥5%%)! 可以进入 GRPO 阶段")
    else:
        print("❌ 未通过 gate (<5%%). 建议检查 SFT 数据质量或调整前缀策略")

    return solve_rate


# ===================== 主入口 =====================

def main():
    parser = argparse.ArgumentParser(description='前缀引导 Warm-Start Pipeline')
    parser.add_argument('--stage', required=True,
                        choices=['prepare', 'generate', 'assemble', 'evaluate'],
                        help='pipeline 阶段')
    parser.add_argument('--tp_size', type=int, default=4,
                        help='vLLM tensor parallel size')
    parser.add_argument('--model_path', type=str, default=None,
                        help='评测使用的模型路径 (evaluate 阶段)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_candidates', type=int, default=4000,
                        help='最大候选题数')
    args = parser.parse_args()

    if args.stage == 'prepare':
        run_prepare(seed=args.seed, max_candidates=args.max_candidates)

    elif args.stage == 'generate':
        run_generate(tp_size=args.tp_size)

    elif args.stage == 'assemble':
        run_assemble(seed=args.seed)

    elif args.stage == 'evaluate':
        mp = args.model_path or os.path.join(BASE_DIR, 'Qwen2.5-Math-7B-WarmStart')
        run_evaluate(model_path=mp, tp_size=args.tp_size)


if __name__ == '__main__':
    main()
