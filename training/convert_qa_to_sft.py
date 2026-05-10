"""
swallowmath_qa 到 SFT sharegpt 格式转换脚本

输入格式（swallowmath_qa 原始数据）：
    {"text": "**Question 1**: 题目...\n**Answer 1**: 解答...\n**Code Implementation 1**: 代码...\n**Question 2**: ..."}

输出格式（LLaMA-Factory sharegpt）：
    {"conversations": [{"from": "human", "value": "题目"}, {"from": "gpt", "value": "解答\n\n答案为 \\boxed{xxx}"}]}

用法:
    python convert_qa_to_sft.py --input_dir ./traindata/swallowmath_qa/stage3-qa/ \
                                --output_path ./sft_data/swallowmath_sft.jsonl \
                                --sample_size 50000
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


def extract_final_answer_from_text(text: str) -> Optional[str]:
    """从答案文本中尝试提取最终数字答案"""
    # 查找 \boxed{} 格式
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # 查找 "答案是" 或 "答案为" 格式
    match = re.search(r'(?:答案(?:是|为))[：:\s]+([^\n,。]+)', text)
    if match:
        return match.group(1).strip()
    
    # 查找最后一个出现的数字/表达式
    numbers = re.findall(r'[-]?\d+(?:\.\d+)?|[a-zA-Z]\^[-]?\d+|sqrt\(\d+\)', text)
    if numbers:
        return numbers[-1].strip()
    
    return None


def ensure_boxed_answer(answer_text: str) -> str:
    """确保答案末尾有 \\boxed{} 格式"""
    # 如果已经有 \boxed{}，直接返回
    if '\\boxed{' in answer_text:
        return answer_text
    
    # 尝试从答案提取最终结果
    final_ans = extract_final_answer_from_text(answer_text)
    if final_ans:
        return answer_text + f"\n\n答案为 $\\boxed{{{final_ans}}}$"
    
    return answer_text


def parse_qa_pairs_from_text(text: str) -> List[Tuple[str, str]]:
    """
    从 swallowmath_qa 格式文本中解析 QA 对
    
    格式：
        **Question 1**: 题目内容
        **Answer 1**: 答案内容
        **Code Implementation 1**: 代码...
        **Question 2**: 题目内容
        ...
    
    返回：[(question, answer), ...]
    """
    qa_pairs = []
    
    # 按 **Question N**: 分割
    question_pattern = r'\*\*Question\s+(\d+)\*\*:\s*'
    answer_pattern = r'\*\*Answer\s+\1\*\*:\s*'
    code_pattern = r'\*\*Code Implementation\s+\1\*\*:\s*'
    
    # 找出所有 Question 块
    question_matches = list(re.finditer(question_pattern, text))
    
    for i, match in enumerate(question_matches):
        q_num = match.group(1)
        q_start = match.end()
        
        # 找对应的 Answer
        answer_match = re.search(answer_pattern.replace(r'\1', q_num), text[q_start:])
        if not answer_match:
            continue
        
        # 提取 Question 内容（从当前位置到 Answer 标记）
        question_text = text[q_start:q_start + answer_match.start()].strip()
        
        # 找对应的 Code Implementation 位置（作为 Answer 的结束位置）
        answer_start = q_start + answer_match.end()
        code_match = re.search(code_pattern.replace(r'\1', q_num), text[answer_start:])
        
        if code_match:
            answer_text = text[answer_start:answer_start + code_match.start()].strip()
        else:
            # 如果没有 Code，就到下一个 Question 或文本末尾
            next_q_match = None
            if i + 1 < len(question_matches):
                next_q_start = question_matches[i + 1].start()
                next_q_match = next_q_start
            
            if next_q_match:
                answer_text = text[answer_start:next_q_match].strip()
            else:
                answer_text = text[answer_start:].strip()
        
        # 去掉末尾可能的 **Question N+1**: 这样的标记
        answer_text = re.sub(r'\*\*Question\s+\d+\*\*:.*$', '', answer_text, flags=re.DOTALL).strip()
        
        if question_text and answer_text:
            # 确保 answer 有 \boxed{} 格式
            answer_text = ensure_boxed_answer(answer_text)
            qa_pairs.append((question_text, answer_text))
    
    return qa_pairs


def convert_file(input_path: str, output_file, max_samples: int = -1, sample_prob: float = 1.0):
    """
    转换单个输入文件
    
    Args:
        input_path: 输入 JSONL 路径
        output_file: 输出文件对象
        max_samples: 最多转换多少条，-1 表示无限制
        sample_prob: 采样概率 (0-1)
    
    Returns:
        (转换成功的 SFT 样本数, 处理的原始样本数)
    """
    sft_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples > 0 and sft_count >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            total_count += 1
            
            # 采样
            if random.random() > sample_prob:
                continue
            
            # 解析 QA 对
            text = item.get('text', '')
            qa_pairs = parse_qa_pairs_from_text(text)
            
            if not qa_pairs:
                continue
            
            # 转换为 sharegpt 格式（每个 QA 对作为一个会话）
            for question, answer in qa_pairs:
                if max_samples > 0 and sft_count >= max_samples:
                    break
                
                sft_item = {
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer}
                    ]
                }
                
                output_file.write(json.dumps(sft_item, ensure_ascii=False) + '\n')
                sft_count += 1
    
    return sft_count, total_count


def main():
    parser = argparse.ArgumentParser(description="swallowmath_qa 转换为 SFT sharegpt 格式")
    parser.add_argument("--input_dir", type=str, 
                       default="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/traindata/swallowmath_qa/stage3-qa/",
                       help="输入目录（包含多个 JSONL 文件）")
    parser.add_argument("--output_path", type=str,
                       default="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/swallowmath_sft.jsonl",
                       help="输出 JSONL 路径")
    parser.add_argument("--sample_size", type=int, default=50000,
                       help="目标采样数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有输入文件
    input_files = sorted(Path(args.input_dir).glob("train-*.jsonl"))
    print(f"[INFO] 找到 {len(input_files)} 个输入文件")
    print(f"[INFO] 目标采样数: {args.sample_size}")
    
    if not input_files:
        print(f"[ERROR] 在 {args.input_dir} 找不到输入文件")
        return
    
    # 计算每个文件的采样概率
    # 先计算总的原始样本数估计
    total_files = len(input_files)
    samples_per_file = args.sample_size // total_files
    
    print(f"\n[INFO] 处理 {len(input_files)} 个文件...")
    
    total_sft_count = 0
    total_original_count = 0
    
    with open(args.output_path, 'w', encoding='utf-8') as output_file:
        for idx, input_file in enumerate(input_files, 1):
            print(f"\n  [{idx}/{len(input_files)}] 处理 {input_file.name}...")
            
            sft_count, orig_count = convert_file(
                str(input_file),
                output_file,
                max_samples=args.sample_size - total_sft_count
            )
            
            print(f"      转换: {sft_count} SFT 样本（来自 {orig_count} 条原始样本）")
            total_sft_count += sft_count
            total_original_count += orig_count
            
            if total_sft_count >= args.sample_size:
                print(f"\n[INFO] 已达到目标样本数 {args.sample_size}，停止处理")
                break
    
    # 汇总统计
    print(f"\n{'='*60}")
    print(f"[DONE] 数据转换完成")
    print(f"{'='*60}")
    print(f"输出路径:        {args.output_path}")
    print(f"总 SFT 样本:     {total_sft_count}")
    print(f"总原始样本:      {total_original_count}")
    print(f"转换率:         {total_sft_count / total_original_count * 100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
