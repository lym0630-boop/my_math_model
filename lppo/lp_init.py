"""
LP 初始化模块 — 从历史数据初始化 P0

功能：
  从 student_responses_*.jsonl 中提取每道题的历史 pass rate，
  作为 LP State Manager 的初始状态 P0。

为什么需要初始化：
  如果 LP state 从零开始（所有题 p=0），前几个 training step 会有大量
  "虚假学习进度"（因为 p 从 0 跳到实际 pass_rate），导致权重分配不合理。
  用历史数据初始化可以跳过这个"冷启动"阶段。

数据来源：
  student_responses_*.jsonl 格式：
    {"question": "...", "ground_truth": "...", "num_correct": 3, "num_total": 8, ...}
  
  其中 num_correct / num_total 就是该题在 DPO-v2 模型上的 pass rate。

与 prepare_math_rlvr_data.py 的关系：
  prepare_math_rlvr_data.py 会为每题分配 sample_id（基于 question hash），
  这里用同样的 hash 逻辑确保 sample_id 一致。
"""

import json
import hashlib
import os
import argparse
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_state_manager import LPStateManager


def compute_sample_id(question: str, ground_truth: str = "") -> str:
    """
    根据题目文本 + 标准答案生成稳定的 sample_id

    使用 (question + "||" + ground_truth) 的 SHA1 前 16 位。
    与 prepare_math_rlvr_data.py 中的 make_sample_id() 逻辑完全一致。

    设计考虑：
      - 加入 ground_truth 是因为同一题目可能有不同 GT 版本
      - SHA1 比 MD5 碰撞率更低
      - 16 位 hex = 64 bit 空间，对 120K 题库足够安全
      - strip() 确保前后空格不影响 hash
    """
    key = question.strip() + "||" + ground_truth.strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def init_lp_state_from_student_responses(
    student_responses_path: str,
    lp_manager: Optional[LPStateManager] = None,
    output_path: Optional[str] = None,
) -> LPStateManager:
    """
    从 student_responses 初始化 LP 状态
    
    Args:
        student_responses_path: student_responses jsonl 文件路径
        lp_manager: 已有的 LPStateManager（可选，不传则新建）
        output_path: 保存初始化后的状态文件路径（可选）
    
    Returns:
        初始化后的 LPStateManager
    
    初始化逻辑：
      - 直接将 student_responses 中的 pass_rate 设为初始 p
      - lp 初始化为 0（无历史变化信息）
      - n_updates 设为 1（标记为已初始化）
    """
    if lp_manager is None:
        lp_manager = LPStateManager()

    if not os.path.exists(student_responses_path):
        print(f"[LP Init] 文件不存在: {student_responses_path}")
        print("[LP Init] 将从零开始（无初始化）")
        return lp_manager

    count = 0
    with open(student_responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            question = d['question']
            ground_truth = d.get('ground_truth', '')
            sample_id = compute_sample_id(question, ground_truth)
            pass_rate = d['num_correct'] / max(d['num_total'], 1)

            # 直接设置初始状态（不经过 EMA 更新）
            lp_manager.states[sample_id] = {
                'p': pass_rate,
                'lp': 0.0,          # 初始无学习进度信息
                'n_updates': 1,     # 标记已初始化
            }
            count += 1

    print(f"[LP Init] 从 student_responses 初始化 {count} 道题的 P0")

    # 打印统计
    summary = lp_manager.get_state_summary()
    print(f"[LP Init] 平均 P0: {summary['avg_p']:.3f}")
    print(f"[LP Init] hard_zero (p<0.05): {summary['hard_zero_count']}")
    print(f"[LP Init] sweet_spot (0.1~0.5): {summary['sweet_spot_count']}")
    print(f"[LP Init] mastered (p>0.8): {summary['mastered_count']}")

    # 可选：保存初始状态
    if output_path:
        lp_manager.save_state(output_path)
        print(f"[LP Init] 已保存初始状态到: {output_path}")

    return lp_manager


def main():
    """命令行入口：独立运行初始化"""
    parser = argparse.ArgumentParser(description='从 student_responses 初始化 LP 状态')
    parser.add_argument('--student_responses',
                        default='/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/sft_data/student_responses_120k_expanded_v4.jsonl',
                        help='student_responses jsonl 文件路径')
    parser.add_argument('--output',
                        default='/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/lppo/lp_state_init.json',
                        help='输出的初始 LP 状态文件')
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--kappa', type=float, default=8.0)
    parser.add_argument('--b', type=float, default=0.5)
    args = parser.parse_args()

    mgr = LPStateManager(beta=args.beta, kappa=args.kappa, b=args.b)
    init_lp_state_from_student_responses(args.student_responses, mgr, args.output)


if __name__ == "__main__":
    main()
