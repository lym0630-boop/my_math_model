#!/bin/bash
# GSM8K Few-Shot 评测：base model 和 CPT checkpoint 并行
# 4 卡机器，每个模型 2 卡，同时跑

set -e

PIPELINE_DIR="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline"
EVAL_SCRIPT="${PIPELINE_DIR}/eval_gsm8k_fewshot.py"

# 模型路径
BASE_MODEL="${PIPELINE_DIR}/Qwen2.5-Math-7B"
CPT_MODEL="${PIPELINE_DIR}/model_eval/checkpoint-3800"

# 结果目录
OUTPUT_DIR="${PIPELINE_DIR}/gsm8k_results"
LOG_DIR="${PIPELINE_DIR}/gsm8k_results/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================================"
echo "  GSM8K 8-Shot 评测（base model 专用）"
echo "  Base: $BASE_MODEL"
echo "  CPT:  $CPT_MODEL"
echo "  GPU 0,1 -> base    GPU 2,3 -> CPT    并行"
echo "============================================================"

# --- base model: GPU 0,1 ---
echo ""
echo "[GPU 0,1] 启动 base model 评测..."
CUDA_VISIBLE_DEVICES=0,1 python3 "$EVAL_SCRIPT" \
    --model_path "$BASE_MODEL" \
    --output_path "${OUTPUT_DIR}/fewshot_base.jsonl" \
    --tensor_parallel_size 2 \
    --num_shots 8 \
    > "${LOG_DIR}/fewshot_base.log" 2>&1 &
PID_BASE=$!
echo "[GPU 0,1] PID: $PID_BASE"

# --- CPT checkpoint: GPU 2,3 ---
echo "[GPU 2,3] 启动 CPT checkpoint 评测..."
CUDA_VISIBLE_DEVICES=2,3 python3 "$EVAL_SCRIPT" \
    --model_path "$CPT_MODEL" \
    --output_path "${OUTPUT_DIR}/fewshot_cpt3800.jsonl" \
    --tensor_parallel_size 2 \
    --num_shots 8 \
    > "${LOG_DIR}/fewshot_cpt3800.log" 2>&1 &
PID_CPT=$!
echo "[GPU 2,3] PID: $PID_CPT"

echo ""
echo "============================================================"
echo "  两个评测已并行启动"
echo "  查看日志:"
echo "    tail -f ${LOG_DIR}/fewshot_base.log"
echo "    tail -f ${LOG_DIR}/fewshot_cpt3800.log"
echo "============================================================"

# 等待完成
FAILED=0

wait $PID_BASE
if [ $? -eq 0 ]; then
    echo "[✓] base model 评测完成"
else
    echo "[✗] base model 评测失败，查看: ${LOG_DIR}/fewshot_base.log"
    FAILED=$((FAILED + 1))
fi

wait $PID_CPT
if [ $? -eq 0 ]; then
    echo "[✓] CPT checkpoint 评测完成"
else
    echo "[✗] CPT checkpoint 评测失败，查看: ${LOG_DIR}/fewshot_cpt3800.log"
    FAILED=$((FAILED + 1))
fi

# 汇总对比
echo ""
echo "============================================================"
echo "  评测结果对比"
echo "============================================================"
python3 -c "
import json, os

results_dir = '${OUTPUT_DIR}'
models = [
    ('Qwen2.5-Math-7B (base)',   'fewshot_base_summary.json'),
    ('checkpoint-3800 (CPT)',     'fewshot_cpt3800_summary.json'),
]

scores = []
for name, fname in models:
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f'  {name:<30} {d[\"accuracy\"]:>6.2f}%  ({d[\"correct\"]}/{d[\"total\"]})')
        scores.append(d['accuracy'])
    else:
        print(f'  {name:<30} 结果文件未找到')

if len(scores) == 2:
    delta = scores[1] - scores[0]
    print(f'  {\"\":<30} -------')
    print(f'  {\"CPT vs Base\":<30} {delta:>+6.2f}%')
"
echo "============================================================"
