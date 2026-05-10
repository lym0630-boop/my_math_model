#!/bin/bash
# DPO 迭代 Pipeline - Round 2
# 运行环境: 4×A100-40GB
# 前置: round2_questions.jsonl 已生成 (16K 题)
#
# 流程:
#   1. Student 推理 (Qwen2.5-Math-7B): 16K × 8 = 128K 次生成, ~2h
#   2. Teacher 推理 (DeepSeek-R1-32B):  ~3K 全错题, ~30min
#   3. 合并 Round 1 + Round 2, 组装最终 DPO pairs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  DPO 迭代 Pipeline - Round 2"
echo "============================================"
echo ""

# GPU 检查
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
echo ""

# Round 2 题数
echo "Round 2 题数: $(wc -l < sft_data/round2_questions.jsonl)"
echo ""

# ---- Step 1: Student 推理 ----
echo "===== Step 1/3: Student 推理 (16K × 8) ====="
python3 dpo_iterative_pipeline.py \
    --stage student_infer \
    --round 2 \
    --tp_size 4

echo ""

# ---- Step 2: Teacher 推理 ----
echo "===== Step 2/3: Teacher 推理 (全错题) ====="
python3 dpo_iterative_pipeline.py \
    --stage teacher_infer \
    --round 2 \
    --tp_size 4

echo ""

# ---- Step 3: 合并组装 ----
echo "===== Step 3/3: 合并 Round 1+2, 组装 DPO pairs ====="
python3 dpo_iterative_pipeline.py \
    --stage assemble \
    --round 2 \
    --target 8000

echo ""
echo "============================================"
echo "  完成！"
echo "============================================"
echo "输出: sft_data/dpo_train_final.jsonl"
echo "文件大小: $(ls -lh sft_data/dpo_train_final.jsonl | awk '{print $5}')"
