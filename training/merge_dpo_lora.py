"""
合并 DPO LoRA 权重 + 清理 tokenizer

LLaMA-Factory 保存的 tokenizer 有问题:
  - 缺少 chat_template (内嵌在 tokenizer_config.json 中的)
  - 缺少 added_tokens_decoder (22个特殊token的定义)
  - 缺少 additional_special_tokens
  - 缺少 add_bos_token

解决方案: 合并后完整复制原始模型的 tokenizer 文件
"""

import os
import shutil
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(base_model_path, adapter_path, output_path):
    print("=" * 50)
    print("Step 1: 合并 LoRA 权重")
    print("=" * 50)
    print("基础模型: %s" % base_model_path)
    print("LoRA adapter: %s" % adapter_path)
    print("输出: %s" % output_path)
    print()

    # 加载基础模型
    print("[1/5] 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA adapter 并合并
    print("[2/5] 加载 LoRA adapter 并合并...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print("  合并完成, 参数量: %.2fB" % (sum(p.numel() for p in model.parameters()) / 1e9))

    # 保存合并后的模型权重
    print("[3/5] 保存模型权重...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    # 清理 tokenizer: 完整复制原始模型的 tokenizer 文件
    print("[4/5] 清理 tokenizer (用原始模型覆盖)...")
    tokenizer_files = [
        'tokenizer.json',
        'tokenizer_config.json',
        'merges.txt',
        'vocab.json',
    ]
    for fname in tokenizer_files:
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print("  复制: %s" % fname)

    # 删除 LLaMA-Factory 生成的多余文件
    for fname in ['chat_template.jinja']:
        fpath = os.path.join(output_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print("  删除: %s (不需要, chat_template 已在 tokenizer_config.json 中)" % fname)

    # 验证
    print()
    print("[5/5] 验证...")
    tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)

    # chat_template
    has_ct = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    print("  chat_template: %s" % ("✅" if has_ct else "❌ 缺失"))

    # special tokens
    print("  eos_token: %s" % tokenizer.eos_token)
    print("  pad_token: %s" % tokenizer.pad_token)
    print("  vocab_size: %d" % tokenizer.vocab_size)

    # 对比原始 tokenizer
    orig_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    test_text = "Please reason step by step, and put your final answer within \\boxed{}."
    orig_ids = orig_tokenizer.encode(test_text)
    new_ids = tokenizer.encode(test_text)
    print("  编码一致性: %s" % ("✅ 完全一致" if orig_ids == new_ids else "❌ 不一致!"))

    # 模型文件大小
    model_files = [f for f in os.listdir(output_path) if f.endswith('.safetensors')]
    total_size = sum(os.path.getsize(os.path.join(output_path, f)) for f in model_files)
    print("  模型大小: %.1f GB (%d 个文件)" % (total_size / 1024**3, len(model_files)))

    print()
    print("✅ 完成: %s" % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/Qwen2.5-Math-7B-Instruct')
    parser.add_argument('--adapter',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/dpo_output')
    parser.add_argument('--output',
        default='/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/Qwen2.5-Math-7B-DPO')
    args = parser.parse_args()

    merge_and_save(args.base_model, args.adapter, args.output)
