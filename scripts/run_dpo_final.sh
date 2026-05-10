#!/bin/bash
# DPO 数据构造 Pipeline（最终版）
# 运行环境: 4×A100-40GB
#
# 新策略:
#   - 24K 题，Student 每题 8 次采样
#   - 答对>=半 → on-policy pair
#   - 答对<半 或 全错 → Teacher 采样 2 次，两次全对才作为 chosen
#
# 预计耗时: Student ~3h + Teacher ~1h = ~4h

set -e
cd "$(dirname "$0")"

# 使用你的 conda 环境
PYTHON="/cfs/cfs-esygraib/belvathliu/cv3/cv3_grpo/bin/python3"

echo "============================================"
echo "  DPO 数据构造 Pipeline（最终版）"
echo "============================================"
echo "  题目数:   24,000"
echo "  Student:  Qwen2.5-Math-7B × 8 采样"
echo "  Teacher:  DeepSeek-R1-32B × 2 采样"
echo "  目标:     ~8K DPO pairs"
echo "============================================"
echo ""

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
$PYTHON -c "import vllm; print('vLLM:', vllm.__version__); import transformers; print('transformers:', transformers.__version__)"
echo ""

# ---- Stage 1: Student 推理 ----
echo "===== Stage 1/3: Student 推理 (24K × 8 = 192K) ====="
$PYTHON dpo_final_pipeline.py --stage student --tp_size 4
echo ""

# ---- Stage 2: Teacher 推理 ----
echo "===== Stage 2/3: Teacher 推理 (2次采样, 4-shot格式对齐) ====="
$PYTHON dpo_final_pipeline.py --stage teacher --tp_size 4
echo ""

# ---- Stage 3: 组装 ----
echo "===== Stage 3/3: 组装 DPO pairs (含长度过滤) ====="
$PYTHON dpo_final_pipeline.py --stage assemble --target 8000
echo ""

echo "============================================"
echo "  完成！"
echo "============================================"
