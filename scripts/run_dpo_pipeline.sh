#!/bin/bash
# DPO 数据构造 Pipeline: Teacher 推理 + 偏好对组装
# 运行环境: 4×A100-40GB
# 前置条件: student_responses_12k.jsonl 已生成

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TEACHER_MODEL="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/DeepSeek-R1-Distill-Qwen-32B"
STUDENT_FILE="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/student_responses_12k.jsonl"
TEACHER_FILE="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/teacher_responses.jsonl"
DPO_OUTPUT="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/sft_data/dpo_train_8k.jsonl"

echo "============================================"
echo "  DPO 数据构造 Pipeline"
echo "============================================"
echo ""

# ---- Step 1: Teacher 推理 ----
echo "===== Step 1/2: Teacher 推理 (DeepSeek-R1-32B) ====="
echo "模型: $TEACHER_MODEL"
echo "输入: $STUDENT_FILE (全错的题目)"
echo ""

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"

# Teacher 推理: 对 Student 全错的 ~3561 题生成 chosen
# TP=4, BF16, greedy decoding
python3 teacher_inference.py \
    --model_path "$TEACHER_MODEL" \
    --input "$STUDENT_FILE" \
    --output "$TEACHER_FILE" \
    --tp_size 4 \
    --max_new_tokens 4096 \
    --batch_size 500

echo ""
echo "Teacher 输出: $TEACHER_FILE"
echo "文件大小: $(ls -lh $TEACHER_FILE | awk '{print $5}')"

# ---- Step 2: 组装 DPO pairs ----
echo ""
echo "===== Step 2/2: 组装 DPO 偏好对 ====="

python3 assemble_dpo_pairs.py \
    --student_file "$STUDENT_FILE" \
    --teacher_file "$TEACHER_FILE" \
    --output "$DPO_OUTPUT" \
    --target 8000

echo ""
echo "============================================"
echo "  完成！"
echo "============================================"
echo "DPO 训练数据: $DPO_OUTPUT"
echo "文件大小: $(ls -lh $DPO_OUTPUT | awk '{print $5}')"
echo ""
echo "下一步: 用这份数据进行 DPO 训练"
