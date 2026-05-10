"""
将 dpo_train_final.jsonl 转成 LLaMA-Factory 支持的 sharegpt preference 格式

输入格式 (我们的):
{
    "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
    "chosen": "推理过程...\boxed{answer}",
    "rejected": "错误推理...\boxed{wrong}",
    ...
}

输出格式 (LLaMA-Factory sharegpt preference):
{
    "conversations": [
        {"from": "human", "value": "{question}"}
    ],
    "chosen": {"from": "gpt", "value": "chosen_response"},
    "rejected": {"from": "gpt", "value": "rejected_response"},
    "system": "Please reason step by step, and put your final answer within \\boxed{}."
}
"""

import json
import re
import argparse


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_question_from_prompt(prompt):
    """从我们的 prompt 格式中提取纯 question 文本"""
    # 格式: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
    m = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt  # fallback


def convert(input_path, output_path, max_length=None):
    """转换数据格式"""
    results = []
    skipped = 0

    with open(input_path) as f:
        for line in f:
            d = json.loads(line)

            question = extract_question_from_prompt(d['prompt'])
            chosen = d['chosen']
            rejected = d['rejected']

            # 可选: 截断超长样本
            if max_length:
                if len(chosen) > max_length:
                    chosen = chosen[:max_length]
                if len(rejected) > max_length:
                    rejected = rejected[:max_length]

            item = {
                "conversations": [
                    {"from": "human", "value": question}
                ],
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": rejected},
                "system": SYSTEM_PROMPT,
            }
            results.append(item)

    # LLaMA-Factory 偏好用 json 数组格式
    with open(output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("转换完成: %d 条" % len(results))
    print("输出: %s" % output_path)
    return len(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/dpo_train_final.jsonl')
    parser.add_argument('--output',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/folder/LLaMA-Factory/data/dpo_math_8k.json')
    parser.add_argument('--max_length', type=int, default=4096,
        help='截断超长样本（字符数）')
    args = parser.parse_args()

    convert(args.input, args.output, args.max_length)
