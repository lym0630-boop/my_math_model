"""
前缀引导 Rollout — 为 hard-zero 题准备带前缀的 prompt

功能：
  当 LP State Manager 检测到某些题目是 "hard-zero"（p < 0.05，模型完全不会），
  为这些题准备带参考解前缀的 prompt，帮助模型从中间步骤开始续写，
  从而获得正向 reward 信号。

与 prefix_guided_warmstart.py 的关系：
  复用 warmstart 中的核心函数：
    - find_semantic_breakpoints(): 找语义断点
    - cut_prefix(): 按比例截取前缀
    - check_answer_leakage(): 检查答案泄露
    - truncate_before_leakage(): 截短泄露前缀
  
  区别：
    - warmstart 是训练前做 SFT 的 pipeline
    - 本模块是训练中动态为 hard-zero 题准备 PG prompt

使用方式：
  在 build_cycle_data.py 中调用，为下一 cycle 的 hard-zero 题
  生成带前缀的 prompt 变体。

设计决策：
  Q: 为什么不直接在 rollout 时拼前缀？
  A: 因为 verl 的 rollout 引擎（vLLM）需要标准化的 prompt 格式，
     在数据准备阶段拼好前缀更安全，不需要改 rollout 代码。
"""

import os
import sys
import json
import random
from typing import Optional

# 复用 warmstart 的核心函数
BASE_DIR = '/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline'
sys.path.insert(0, BASE_DIR)
from prefix_guided_warmstart import (
    find_semantic_breakpoints,
    cut_prefix,
    check_answer_leakage,
    truncate_before_leakage,
)


# 前缀比例配置：随机三档，模拟不同程度的"脚手架"
# short：只给一点开头提示，考验模型能否自己推理
# mid：给到关键步骤前，帮模型找到方向
# long：给到接近结尾，主要考验最后计算/整理步骤
PG_PREFIX_RATIOS = {
    'short': (0.15, 0.25),  # 50% 概率
    'mid':   (0.25, 0.40),  # 35% 概率
    'long':  (0.40, 0.55),  # 15% 概率
}
PG_RATIO_PROBS = [0.50, 0.35, 0.15]  # 对应 short, mid, long
PG_RATIO_NAMES = ['short', 'mid', 'long']

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def _sample_prefix_ratio() -> tuple:
    """
    按概率随机选择 prefix ratio 档位

    分布设计原理：
      - short (15-25%) 50%：大多数时候只给很少提示，迫使模型学自主推理
      - mid (25-40%) 35%：给到关键步骤，帮模型找到突破口
      - long (40-55%) 15%：少量长前缀，用于最难的题只需要学最后一步

    Returns:
        (ratio: float, level_name: str)
    """
    level = random.choices(PG_RATIO_NAMES, weights=PG_RATIO_PROBS, k=1)[0]
    lo, hi = PG_PREFIX_RATIOS[level]
    ratio = random.uniform(lo, hi)
    return ratio, level


def prepare_pg_prompt(
    question: str,
    reference_answer: str,
    ground_truth: str,
    prefix_ratio: float = None,
) -> Optional[dict]:
    """
    为单道 hard-zero 题准备带前缀的 prompt

    Args:
        question: 题目文本
        reference_answer: 参考解（来自 Teacher 模型或 dpo_questions）
        ground_truth: 标准答案（用于泄露检查）
        prefix_ratio: 前缀截取比例。None 时按随机三档采样。

    Returns:
        dict: 带前缀的 prompt 记录，格式与 verl parquet 兼容
              如果无法生成有效前缀，返回 None

    处理流程：
      1. 选择 prefix ratio（随机三档或指定）
      2. 找语义断点
      3. 按比例截取前缀
      4. 答案泄露检查 → 如有泄露则截短
      5. 组装为 verl prompt 格式（前缀拼在 assistant 内容中）
    """
    # Step 1: 选择 prefix ratio
    if prefix_ratio is None:
        prefix_ratio, ratio_level = _sample_prefix_ratio()
    else:
        ratio_level = 'custom'

    # Step 2: 找断点
    breakpoints = find_semantic_breakpoints(reference_answer)
    if not breakpoints:
        return None

    # Step 3: 截取前缀
    prefix = cut_prefix(reference_answer, prefix_ratio, breakpoints)
    if prefix is None or len(prefix) < 30:
        return None

    # Step 4: 答案泄露检查
    if check_answer_leakage(prefix, ground_truth):
        bp_in_prefix = [bp for bp in breakpoints if bp <= len(prefix)]
        prefix = truncate_before_leakage(prefix, ground_truth, bp_in_prefix)
        if len(prefix) < 30:
            return None

    # Step 5: 组装 prompt
    # 关键设计：前缀放在 assistant 的 content 中
    # verl 的 rollout 会从 assistant content 之后开始生成
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefix},  # ← 前缀在这里
    ]

    return {
        "prompt": prompt,
        "prefix": prefix,
        "prefix_ratio": prefix_ratio,
        "prefix_level": ratio_level,  # short/mid/long
        "prefix_len": len(prefix),
        "ref_answer_len": len(reference_answer),
    }


def batch_prepare_pg_prompts(
    hard_zero_items: list,
    reference_answers: dict,
    max_pg_samples: int = 500,
    seed: int = 42,
) -> list:
    """
    批量为 hard-zero 题目准备 PG prompts
    
    Args:
        hard_zero_items: LP state 中 p < 0.05 的题目列表
                        每个元素: {"sample_id": str, "question": str, "ground_truth": str}
        reference_answers: {question_key: answer_text} 参考解字典
        max_pg_samples: 最大 PG 样本数（控制 PG 样本占比）
        seed: 随机种子
    
    Returns:
        list of dict: 可直接追加到训练 parquet 的记录列表
    """
    random.seed(seed)
    random.shuffle(hard_zero_items)

    pg_records = []
    stats = {'total': 0, 'no_ref': 0, 'prefix_fail': 0, 'success': 0}

    for item in hard_zero_items:
        if len(pg_records) >= max_pg_samples:
            break

        stats['total'] += 1
        question = item['question']
        qk = question[:120]

        # 找参考解
        ref_answer = reference_answers.get(qk)
        if not ref_answer:
            stats['no_ref'] += 1
            continue

        # 生成 PG prompt
        result = prepare_pg_prompt(
            question=question,
            reference_answer=ref_answer,
            ground_truth=item['ground_truth'],
        )

        if result is None:
            stats['prefix_fail'] += 1
            continue

        # 组装完整记录
        pg_records.append({
            "data_source": "math",
            "prompt": result["prompt"],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": item['ground_truth'],
            },
            "extra_info": {
                "sample_id": item['sample_id'],
                "is_pg_sample": True,       # 标记这是 PG 样本
                "prefix_ratio": result["prefix_ratio"],
                "prefix_level": result["prefix_level"],  # short/mid/long
                "prefix_len": result["prefix_len"],
            },
        })
        stats['success'] += 1

    print(f"[PG Rollout] 统计: {stats}")
    return pg_records


if __name__ == "__main__":
    # 简单测试
    test_question = "Find the value of x if 2x + 3 = 7"
    test_answer = """Let me solve this step by step.

We have the equation: 2x + 3 = 7

**Step 1:** Subtract 3 from both sides.
2x + 3 - 3 = 7 - 3
2x = 4

**Step 2:** Divide both sides by 2.
x = 4/2
x = 2

Therefore, the answer is $\\boxed{2}$."""

    result = prepare_pg_prompt(test_question, test_answer, "2")
    if result:
        print("✅ PG prompt 生成成功")
        print(f"  前缀长度: {result['prefix_len']} / {result['ref_answer_len']}")
        print(f"  前缀比例: {result['prefix_ratio']}")
        print(f"  前缀内容: {result['prefix'][:100]}...")
    else:
        print("❌ PG prompt 生成失败")
