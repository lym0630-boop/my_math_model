"""
将 LoRA 权重合并回完整模型

用法:
    python merge_lora_weights.py \
        --model_name_or_path /path/to/checkpoint-3800 \
        --adapter_name_or_path /path/to/sft_output \
        --output_dir /path/to/merged_model

功能:
    1. 加载基础模型和 LoRA 权重
    2. 合并 LoRA 权重到模型
    3. 保存合并后的模型（约 15GB）
"""

import argparse
import os
import json
from pathlib import Path


def merge_lora(model_path: str, lora_path: str, output_dir: str):
    """合并 LoRA 权重到基础模型"""
    
    print("="*60)
    print("合并 LoRA 权重到模型")
    print("="*60)
    print(f"基础模型:    {model_path}")
    print(f"LoRA 权重:   {lora_path}")
    print(f"输出目录:    {output_dir}")
    print("="*60)
    
    # 加载模型和 LoRA
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("[ERROR] 需要安装 peft 库: pip install peft")
        return False
    
    try:
        print("\n[1/3] 加载 LoRA 模型...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        
        print("[2/3] 合并 LoRA 权重...")
        merged_model = model.merge_and_unload()
        
        print("[3/3] 保存合并后的模型...")
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        
        # 也保存 tokenizer
        print("[4/4] 保存 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        
        print("\n" + "="*60)
        print("[✓] 合并完成")
        print("="*60)
        print(f"输出模型: {output_dir}")
        print(f"文件大小: {os.path.getsize(os.path.join(output_dir, 'model.safetensors')) / 1e9:.2f} GB")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 合并失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到模型")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="基础模型路径（CPT checkpoint）")
    parser.add_argument("--adapter_name_or_path", type=str, required=True,
                       help="LoRA 权重路径（SFT 输出）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="合并后模型保存路径")
    args = parser.parse_args()
    
    success = merge_lora(
        args.model_name_or_path,
        args.adapter_name_or_path,
        args.output_dir
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
