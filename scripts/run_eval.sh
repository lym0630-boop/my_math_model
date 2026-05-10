#!/bin/bash
# ============================================================
# 批量评测脚本 - 8 卡 A100 同时跑多个模型
# 用法: bash run_eval.sh
# ============================================================

# 评测任务配置目录
TASK_DIR="./eval_tasks"
TASKS="gsm8k_local,math_local"

# ====== 在这里配置你要评测的模型 ======
# 格式: "模型路径|结果保存名"
# 最多 8 个（对应 8 张卡）
MODELS=(
    "./Qwen2.5-Math-7B|baseline"
    # 训练后的模型，取消注释并修改路径：
    # "/cfs/xxx/finetuned_v1|finetuned_v1"
    # "/cfs/xxx/finetuned_v2|finetuned_v2"
    # "/cfs/xxx/checkpoint-500|ckpt_500"
    # "/cfs/xxx/checkpoint-1000|ckpt_1000"
    # "/cfs/xxx/checkpoint-1500|ckpt_1500"
    # "/cfs/xxx/checkpoint-2000|ckpt_2000"
    # "/cfs/xxx/checkpoint-2500|ckpt_2500"
)

# 结果保存根目录
OUTPUT_ROOT="./eval_results"
mkdir -p "$OUTPUT_ROOT"

# 日志目录
LOG_DIR="./eval_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  批量评测启动"
echo "  模型数量: ${#MODELS[@]}"
echo "  评测任务: $TASKS"
echo "============================================================"

PIDS=()

for i in "${!MODELS[@]}"; do
    # 解析模型路径和保存名
    IFS='|' read -r MODEL_PATH SAVE_NAME <<< "${MODELS[$i]}"

    GPU_ID=$i
    OUTPUT_PATH="${OUTPUT_ROOT}/${SAVE_NAME}"
    LOG_FILE="${LOG_DIR}/${SAVE_NAME}.log"

    echo ""
    echo "[GPU $GPU_ID] 模型: $MODEL_PATH"
    echo "[GPU $GPU_ID] 结果: $OUTPUT_PATH"
    echo "[GPU $GPU_ID] 日志: $LOG_FILE"

    # 每个模型分配一张卡，后台运行
    CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval \
        --model vllm \
        --model_args pretrained=$MODEL_PATH,trust_remote_code=True,gpu_memory_utilization=0.85 \
        --tasks $TASKS \
        --include_path $TASK_DIR \
        --batch_size auto \
        --output_path $OUTPUT_PATH \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    echo "[GPU $GPU_ID] 已启动 (PID: ${PIDS[$i]})"
done

echo ""
echo "============================================================"
echo "  所有模型评测已启动，等待完成..."
echo "  查看日志: tail -f ${LOG_DIR}/<模型名>.log"
echo "============================================================"

# 等待所有任务完成
FAILED=0
for i in "${!PIDS[@]}"; do
    IFS='|' read -r MODEL_PATH SAVE_NAME <<< "${MODELS[$i]}"
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[✓] $SAVE_NAME 评测完成"
    else
        echo "[✗] $SAVE_NAME 评测失败 (exit code: $EXIT_CODE)，查看日志: ${LOG_DIR}/${SAVE_NAME}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  全部完成！成功: $((${#MODELS[@]} - FAILED))  失败: $FAILED"
echo "  结果目录: $OUTPUT_ROOT"
echo "============================================================"

# 汇总所有结果
echo ""
echo "==================== 评测结果汇总 ===================="
for i in "${!MODELS[@]}"; do
    IFS='|' read -r MODEL_PATH SAVE_NAME <<< "${MODELS[$i]}"
    RESULT_FILE=$(find "${OUTPUT_ROOT}/${SAVE_NAME}" -name "results.json" 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ]; then
        echo ""
        echo "--- $SAVE_NAME ---"
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
results = data.get('results', {})
for task, metrics in results.items():
    for k, v in metrics.items():
        if 'exact_match' in k or 'acc' in k:
            print(f'  {task}: {k} = {v}')
" 2>/dev/null || echo "  结果解析失败，请手动查看: $RESULT_FILE"
    fi
done
echo "======================================================"
