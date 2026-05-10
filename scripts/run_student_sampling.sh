#!/bin/bash
# Student Rejection Sampling 启动脚本
# 运行环境: 4×A100-40GB
# 预计耗时: 约 1~2 小时

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/Qwen2.5-Math-7B-Instruct"
INPUT_FILE="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/dpo_questions_12k.jsonl"
OUTPUT_FILE="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/student_responses_12k.jsonl"

echo "============================================"
echo "  Student Rejection Sampling for DPO"
echo "============================================"
echo "模型:     $MODEL_PATH"
echo "输入:     $INPUT_FILE (12K 题目)"
echo "输出:     $OUTPUT_FILE"
echo "配置:     4×A100-40GB, TP=4, n=8, temp=0.7"
echo "预计:     96K 次生成, 约 1~2 小时"
echo "============================================"

# 检查 GPU
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

# 检查 vllm
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"

# 检查输入文件
NUM_QUESTIONS=$(wc -l < "$INPUT_FILE")
echo "输入题目数: $NUM_QUESTIONS"
echo ""

# 运行
echo "开始推理..."
python3 student_rejection_sampling.py \
    --model_path "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --n_samples 8 \
    --tp_size 4 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --batch_size 2000

echo ""
echo "完成！输出文件: $OUTPUT_FILE"
echo "文件大小: $(ls -lh $OUTPUT_FILE | awk '{print $5}')"
