"""
validator.py — 数学感知校验

在 LLM 处理后检查数学内容是否被洗坏。
不做复杂公式 AST 解析，只做轻量一致性检查。
"""
from __future__ import annotations

import re
from typing import List, Set

from .config import PipelineConfig
from .scorer import BaseScorer


# ===========================================================================
# 提取函数
# ===========================================================================

# 数字：整数、小数、分数、百分比
_NUMBER_EXTRACT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?%?")

# 变量模式：单字母 + 可选下标
_VARIABLE_EXTRACT_RE = re.compile(r"\b([a-zA-Z])(?:_\{?([a-zA-Z0-9]+)\}?)?\b")

# 常见停用词
_STOP_WORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "is", "it",
    "by", "as", "or", "if", "so", "no", "be", "do", "we", "he", "me",
    "up", "am", "my", "I",
})

# 数学 Unicode 符号
_MATH_SYMBOL_RE = re.compile(r"[+\-*/=≠≤≥≈∞∑∏∫∂∇±×÷∈∉⊂⊃∪∩∧∨¬⇒⇔∀∃]")

# LaTeX 命令
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")

# 括号类型
_BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]


def extract_numbers(text: str) -> List[str]:
    """提取文本中的所有数字（含小数、分数、百分比）"""
    return _NUMBER_EXTRACT_RE.findall(text)


def extract_variables(text: str) -> Set[str]:
    """
    提取文本中的数学变量。
    返回变量名集合（如 {'x', 'y', 'a_n', 'x_i'}）。
    过滤掉停用词。
    """
    variables = set()
    for m in _VARIABLE_EXTRACT_RE.finditer(text):
        var = m.group(1)
        subscript = m.group(2)
        if var.lower() in _STOP_WORDS:
            continue
        if subscript:
            variables.add(f"{var}_{subscript}")
        else:
            variables.add(var)
    return variables


def count_math_symbols(text: str) -> int:
    """统计数学 Unicode 符号数量"""
    return len(_MATH_SYMBOL_RE.findall(text))


def count_latex_commands(text: str) -> int:
    """统计 LaTeX 命令数量"""
    return len(_LATEX_CMD_RE.findall(text))


# ===========================================================================
# 检查函数
# ===========================================================================

def check_number_recall(raw_text: str, processed_text: str) -> float:
    """
    检查数字保留率。

    返回 recall = |交集| / |原文数字集|
    原文无数字时返回 1.0
    """
    raw_nums = extract_numbers(raw_text)
    if not raw_nums:
        return 1.0
    post_nums = extract_numbers(processed_text)
    raw_set = set(raw_nums)
    post_set = set(post_nums)
    intersection = raw_set & post_set
    return len(intersection) / len(raw_set)


def check_variable_recall(raw_text: str, processed_text: str) -> float:
    """
    检查变量保留率。

    返回 recall = |交集| / |原文变量集|
    原文无变量时返回 1.0
    """
    raw_vars = extract_variables(raw_text)
    if not raw_vars:
        return 1.0
    post_vars = extract_variables(processed_text)
    intersection = raw_vars & post_vars
    return len(intersection) / len(raw_vars)


def check_symbol_recall(raw_text: str, processed_text: str) -> float:
    """
    检查数学符号保留率。

    只看数量比例，不要求完全相同。
    原文无符号时返回 1.0
    """
    raw_count = count_math_symbols(raw_text)
    if raw_count == 0:
        return 1.0
    post_count = count_math_symbols(processed_text)
    return min(post_count / raw_count, 1.5)  # 允许少量增加，cap 在 1.5


def check_bracket_balance(raw_text: str, processed_text: str) -> bool:
    """
    检查括号平衡是否变差。

    对每种括号，计算 |左-右| 的差值。
    如果处理后的不平衡度大于原文，认为变差。
    """
    for left, right in _BRACKET_PAIRS:
        raw_imbalance = abs(raw_text.count(left) - raw_text.count(right))
        post_imbalance = abs(processed_text.count(left) - processed_text.count(right))
        # 允许处理后改善平衡，但不能变差太多
        if post_imbalance > raw_imbalance + 2:
            return False
    return True


def check_latex_balance(raw_text: str, processed_text: str) -> bool:
    """
    检查 LaTeX 环境平衡是否变差。

    检查：
    1. $ 数量奇偶性不变差
    2. \\begin 和 \\end 数量差不变差
    """
    # $ 符号奇偶性
    raw_dollar = raw_text.count("$")
    post_dollar = processed_text.count("$")
    # 如果原文 $ 数量是偶数（平衡），处理后变成奇数（不平衡），则失败
    if raw_dollar % 2 == 0 and post_dollar % 2 != 0:
        return False

    # \begin / \end 平衡
    raw_begin = len(re.findall(r"\\begin\{", raw_text))
    raw_end = len(re.findall(r"\\end\{", raw_text))
    post_begin = len(re.findall(r"\\begin\{", processed_text))
    post_end = len(re.findall(r"\\end\{", processed_text))

    raw_diff = abs(raw_begin - raw_end)
    post_diff = abs(post_begin - post_end)
    if post_diff > raw_diff + 1:
        return False

    return True


def check_length_ratio(raw_text: str, processed_text: str) -> float:
    """计算长度压缩比"""
    raw_len = len(raw_text)
    if raw_len == 0:
        return 1.0
    return len(processed_text) / raw_len


# ===========================================================================
# 综合校验
# ===========================================================================

from .schemas import ValidationResult


def validate(
    raw_text: str,
    processed_text: str,
    score_raw: float,
    scorer: BaseScorer,
    config: PipelineConfig | None = None,
) -> ValidationResult:
    """
    对 LLM 处理结果做全面的数学感知校验。

    参数:
        raw_text: 轻清洗后的原文
        processed_text: LLM 处理后的文本
        score_raw: 原文的 classifier 分数
        scorer: 评分器（用于复评分）
        config: 配置

    返回:
        ValidationResult
    """
    if config is None:
        config = PipelineConfig()

    fail_reasons: list[str] = []

    # A. 数字保留
    number_recall = check_number_recall(raw_text, processed_text)
    if number_recall < config.number_recall_min:
        fail_reasons.append(f"number_recall={number_recall:.2f}<{config.number_recall_min}")

    # B. 变量保留
    variable_recall = check_variable_recall(raw_text, processed_text)
    if variable_recall < config.variable_recall_min:
        fail_reasons.append(f"variable_recall={variable_recall:.2f}<{config.variable_recall_min}")

    # C. 符号保留
    symbol_recall = check_symbol_recall(raw_text, processed_text)
    if symbol_recall < config.symbol_recall_min:
        fail_reasons.append(f"symbol_recall={symbol_recall:.2f}<{config.symbol_recall_min}")

    # D. 括号平衡
    bracket_ok = check_bracket_balance(raw_text, processed_text)
    if not bracket_ok:
        fail_reasons.append("bracket_balance_worsened")

    # E. LaTeX 平衡
    latex_ok = check_latex_balance(raw_text, processed_text)
    if not latex_ok:
        fail_reasons.append("latex_balance_worsened")

    # F. 长度压缩比
    length_ratio = check_length_ratio(raw_text, processed_text)
    if length_ratio < config.length_ratio_min:
        fail_reasons.append(f"length_ratio={length_ratio:.2f}<{config.length_ratio_min}")
    if length_ratio > config.length_ratio_max:
        fail_reasons.append(f"length_ratio={length_ratio:.2f}>{config.length_ratio_max}")

    # G. 复评分
    score_post = scorer.score(processed_text)
    score_drop = score_raw - score_post
    if score_drop > config.max_clean_score_drop and score_post < config.min_post_score_absolute:
        fail_reasons.append(
            f"score_dropped: {score_raw:.2f}->{score_post:.2f} "
            f"(drop={score_drop:.2f}>{config.max_clean_score_drop})"
        )

    validation_pass = len(fail_reasons) == 0

    return ValidationResult(
        validation_pass=validation_pass,
        number_recall=round(number_recall, 4),
        variable_recall=round(variable_recall, 4),
        symbol_recall=round(symbol_recall, 4),
        length_ratio=round(length_ratio, 4),
        bracket_balance_ok=bracket_ok,
        latex_balance_ok=latex_ok,
        score_raw=round(score_raw, 4),
        score_post=round(score_post, 4),
        fail_reasons=fail_reasons,
    )
