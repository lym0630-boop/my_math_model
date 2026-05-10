"""
数学 RLVR Reward Function（含 OpenWebMath CoT 质量奖励）

组合三个信号：
  1. correctness_reward: 答案正确性（\\boxed{} 匹配 ground_truth）
     - 对 → +1.0
     - 错但有 \\boxed{} → -0.9
     - 错且无 \\boxed{} → -1.0
  2. fm_reward: OpenWebMath classifier 对 CoT 文本的质量打分
     - 只在 correct / near-correct 样本上生效，避免抬高“写得像样但答案错”的输出
     - 门控版: max(0, (score - 3) / 2)，score≥3 才有正向奖励
  3. repetition_penalty: 只惩罚明显的循环退化输出
     - 检测重复 n-gram 和重复长行

verl 调用签名:
  compute_score(data_source, solution_str, ground_truth, extra_info) -> dict

Usage:
  custom_reward_function.path=reward_math_rlvr.py
  custom_reward_function.name=compute_score

环境变量控制:
  LAMBDA_FM: OpenWebMath 奖励权重，默认 0.05
  FM_MODEL_PATH: OpenWebMath classifier 模型路径，默认 HuggingFaceTB/openwebmath-classifier
  FM_GATE_THRESHOLD: 门控阈值，默认 3.0（score >= 3 才有正向奖励）
  WRONG_BOXED_PENALTY: 答错但有 \\boxed{} 的惩罚，默认 -0.9
  WRONG_UNFORMATTED_PENALTY: 答错且没有 \\boxed{} 的惩罚，默认 -1.0
  NEAR_CORRECT_REL_TOL: near-correct 相对误差阈值，默认 0.01
  NEAR_CORRECT_ABS_TOL: near-correct 绝对误差阈值，默认 1e-4
  MAX_REPETITION_PENALTY: 最大 repetition 惩罚，默认 0.05
"""

import os
import re
import threading
from fractions import Fraction
from typing import Optional, Tuple


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
    (r"\ ", ""), (" ", ""), ("mbox", "text"),
    (",\\text{and}", ","), ("\\text{and}", ","),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches",
    "hours", "km", "units", "points", "feet", "minutes",
    "digits", "cents", "degrees", "cm", "pounds", "meters",
    "\\ldots", "\\dots", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left):-1]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    final_answer = final_answer.replace("\\dfrac", "\\frac")
    final_answer = final_answer.replace("\\tfrac", "\\frac")
    final_answer = final_answer.replace("\\left", "")
    final_answer = final_answer.replace("\\right", "")
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def parse_numeric_answer(answer: str) -> Optional[float]:
    """将整数/小数/分数（含 \\frac）解析为数值。"""
    normalized = normalize_final_answer(answer)

    if re.fullmatch(r"-?\d+", normalized):
        return float(int(normalized))

    if re.fullmatch(r"-?\d+\.\d+", normalized):
        return float(normalized)

    if re.fullmatch(r"-?\d+/\d+", normalized):
        # 防止分母为 0 导致 ZeroDivisionError
        num, den = normalized.rsplit("/", 1)
        if int(den) == 0:
            return None
        return float(Fraction(normalized))

    frac_match = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", normalized)
    if frac_match:
        sign = -1 if frac_match.group(1) == "-" else 1
        numerator = int(frac_match.group(2))
        denominator = int(frac_match.group(3))
        if denominator == 0:
            return None
        return float(sign * Fraction(numerator, denominator))

    return None


LAMBDA_FM = float(os.environ.get("LAMBDA_FM", "0.05"))
FM_MODEL_PATH = os.environ.get("FM_MODEL_PATH", "HuggingFaceTB/openwebmath-classifier")
FM_GATE_THRESHOLD = float(os.environ.get("FM_GATE_THRESHOLD", "3.0"))
# 适当收窄“错但格式好”和“错且没格式”的差距，避免模型过度偏好自信地 box 错答案。
WRONG_BOXED_PENALTY = float(os.environ.get("WRONG_BOXED_PENALTY", "-0.9"))
WRONG_UNFORMATTED_PENALTY = float(os.environ.get("WRONG_UNFORMATTED_PENALTY", "-1.0"))
NEAR_CORRECT_REL_TOL = float(os.environ.get("NEAR_CORRECT_REL_TOL", "0.01"))
NEAR_CORRECT_ABS_TOL = float(os.environ.get("NEAR_CORRECT_ABS_TOL", "1e-4"))
MAX_REPETITION_PENALTY = float(os.environ.get("MAX_REPETITION_PENALTY", "0.05"))
FORMAT_REWARD_VALUE = max(0.0, WRONG_BOXED_PENALTY - WRONG_UNFORMATTED_PENALTY)


def compute_correctness(solution_str: str, ground_truth: str) -> dict:
    """答案正确性检查。"""
    solution_tail = solution_str[-300:]
    boxed = last_boxed_only_string(solution_tail)
    has_boxed = boxed is not None

    if boxed is not None:
        pred = normalize_final_answer(remove_boxed(boxed))
    else:
        match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution_str)
        pred = normalize_final_answer(match[-1]) if match else "[INVALID]"

    gt = normalize_final_answer(ground_truth)
    pred_num = parse_numeric_answer(pred)
    gt_num = parse_numeric_answer(gt)

    correct = pred == gt
    if not correct and pred_num is not None and gt_num is not None:
        correct = abs(pred_num - gt_num) <= 1e-8

    near_correct = False
    if not correct and pred_num is not None and gt_num is not None:
        abs_err = abs(pred_num - gt_num)
        rel_err = abs_err / max(abs(gt_num), 1.0)
        near_correct = abs_err <= NEAR_CORRECT_ABS_TOL or rel_err <= NEAR_CORRECT_REL_TOL

    outcome_reward = 1.0 if correct else 0.0
    incorrect_penalty = 0.0 if correct else WRONG_UNFORMATTED_PENALTY
    format_reward = 0.0 if correct else (FORMAT_REWARD_VALUE if has_boxed else 0.0)
    score = outcome_reward + incorrect_penalty + format_reward

    return {
        "correctness_score": score,
        "acc": correct,
        "near_correct": near_correct,
        "pred": pred,
        "has_boxed": has_boxed,
        "outcome_reward": outcome_reward,
        "format_reward": format_reward,
        "incorrect_penalty": incorrect_penalty,
        "parseable_answer": pred != "[INVALID]",
    }


_fm_lock = threading.Lock()
_fm_model = None
_fm_tokenizer = None


def _load_fm_model():
    """懒加载 OpenWebMath classifier（首次调用时加载，线程安全）。"""
    global _fm_model, _fm_tokenizer
    if _fm_model is not None:
        return

    with _fm_lock:
        if _fm_model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"[OpenWebMath] 加载模型: {FM_MODEL_PATH}")
        _fm_tokenizer = AutoTokenizer.from_pretrained(FM_MODEL_PATH)
        _fm_model = AutoModelForSequenceClassification.from_pretrained(FM_MODEL_PATH)
        _fm_model.eval()

        if torch.cuda.is_available():
            try:
                _fm_model = _fm_model.half().cuda()
                print("[OpenWebMath] 已加载到 GPU (fp16)")
            except Exception:
                print("[OpenWebMath] GPU 加载失败，使用 CPU")
        else:
            print("[OpenWebMath] 使用 CPU")


def compute_fm_score(cot_text: str) -> Tuple[float, float]:
    """对 CoT 文本打 OpenWebMath 质量分，返回 (raw_score, gated_reward)。"""
    if LAMBDA_FM <= 0:
        return 0.0, 0.0

    _load_fm_model()

    import torch

    inputs = _fm_tokenizer(
        cot_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    device = next(_fm_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _fm_model(**inputs)
        raw_score = outputs.logits.squeeze(-1).float().cpu().item()

    fm_reward = max(0.0, (raw_score - FM_GATE_THRESHOLD) / 2.0)
    return raw_score, fm_reward


def compute_repetition_penalty(text: str) -> float:
    """只惩罚明显的循环退化输出，避免伤到正常数学推理重复。"""
    penalty = 0.0
    tokens = re.findall(r"\S+", text.lower())

    if len(tokens) >= 48:
        max_repetitions = 1
        for n in (6, 8):
            for i in range(len(tokens) - 2 * n + 1):
                span = tokens[i : i + n]
                repetitions = 1
                j = i + n
                while j + n <= len(tokens) and tokens[j : j + n] == span:
                    repetitions += 1
                    j += n
                if repetitions > max_repetitions:
                    max_repetitions = repetitions

        if max_repetitions >= 2:
            penalty += min(0.04, 0.02 * (max_repetitions - 1))

    lines = [line.strip().lower() for line in text.splitlines() if len(line.strip()) >= 20]
    if lines:
        duplicate_lines = len(lines) - len(set(lines))
        if duplicate_lines > 0:
            penalty += min(0.03, 0.01 * duplicate_lines)

    return min(penalty, MAX_REPETITION_PENALTY)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """组合 reward: correctness + OpenWebMath CoT 质量 - repetition_penalty。"""
    corr = compute_correctness(solution_str, ground_truth)

    fm_reward = 0.0
    raw_fm_score = 0.0
    if LAMBDA_FM > 0:
        try:
            raw_fm_score, fm_reward = compute_fm_score(solution_str)
        except Exception as e:
            print(f"[OpenWebMath] 打分失败: {e}")
            fm_reward = 0.0
            raw_fm_score = 0.0

    fm_bonus = 0.0
    if corr["acc"] or corr["near_correct"]:
        fm_bonus = LAMBDA_FM * fm_reward

    repetition_penalty = compute_repetition_penalty(solution_str)
    total_reward_before_penalty = corr["correctness_score"] + fm_bonus
    total_reward = total_reward_before_penalty - repetition_penalty

    return {
        "score": total_reward,
        "total_reward": total_reward,
        "acc": corr["acc"],
        "near_correct": corr["near_correct"],
        "pred": corr["pred"],
        "has_boxed": corr["has_boxed"],
        "correctness_reward": corr["correctness_score"],
        "base_reward": corr["correctness_score"],
        "outcome_reward": corr["outcome_reward"],
        "format_reward": corr["format_reward"],
        "incorrect_penalty": corr["incorrect_penalty"],
        "parseable_answer": corr["parseable_answer"],
        "fm_reward": fm_reward,
        "openwebmath_reward": fm_reward,
        "fm_raw_score": raw_fm_score,
        "fm_bonus": fm_bonus,
        "openwebmath_bonus": fm_bonus,
        "repetition_penalty": repetition_penalty,
        "total_reward_before_penalty": total_reward_before_penalty,
        "repetition_penalized": repetition_penalty > 0,
    }
