#!/bin/bash
# ============================================================
# Qwen2.5-Math SFT LoRA 训练脚本
# ============================================================
# 硬件：4×A100 40GB
# 训练数据：swallowmath_sft（50k 样本）
# 基础模型：checkpoint-3800（CPT 后）
# 输出模型：sft_output
# ============================================================

set -e

PIPELINE_DIR="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline"
LLAMA_FACTORY_DIR="/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/folder/LLaMA-Factory"
CONFIG_FILE="${PIPELINE_DIR}/qwen25_math_sft.yaml"
LOG_DIR="${PIPELINE_DIR}/logs_sft"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Qwen2.5-Math SFT LoRA 训练"
echo "============================================================"
echo "  基础模型: checkpoint-3800（CPT 后）"
echo "  数据集:   swallowmath_sft (50k 样本)"
echo "  配置:     ${CONFIG_FILE}"
echo "  输出:     ${PIPELINE_DIR}/model_eval/sft_output"
echo "  硬件:     4×A100 40GB"
echo "============================================================"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查 LLaMA-Factory
if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "[ERROR] LLaMA-Factory 目录不存在: $LLAMA_FACTORY_DIR"
    exit 1
fi

# 检查数据文件
SFT_DATA="${PIPELINE_DIR}/sft_data/swallowmath_sft.jsonl"
if [ ! -f "$SFT_DATA" ]; then
    echo "[ERROR] SFT 数据文件不存在: $SFT_DATA"
    exit 1
fi
echo "[✓] 数据文件存在: $(wc -l < $SFT_DATA) 条样本"

# 检查模型
MODEL_PATH="${PIPELINE_DIR}/model_eval/checkpoint-3800"
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] 模型目录不存在: $MODEL_PATH"
    exit 1
fi
echo "[✓] 模型目录存在: $MODEL_PATH"
echo ""

# 启动 SFT 训练
echo "启动 SFT LoRA 训练..."
cd "$LLAMA_FACTORY_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m llamafactory.train "$CONFIG_FILE" \
    2>&1 | tee "$LOG_DIR/sft_training.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  [✓] 训练完成"
    echo "============================================================"
    
    # 显示输出模型信息
    OUTPUT_DIR="${PIPELINE_DIR}/model_eval/sft_output"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  输出模型: $OUTPUT_DIR"
        ls -lh "$OUTPUT_DIR" | grep -E "adapter|config|pytorch" | head -5
    fi
    
    echo ""
    echo "  下一步："
    echo "    1. 合并 LoRA 权重到完整模型"
    echo "       python3 -c \"from peft import AutoPeftModelForCausalLM; ...\""
    echo "    2. 评测对比"
    echo "       python eval_math.py --model $OUTPUT_DIR --dataset gsm8k"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "  [✗] 训练失败 (exit code: $EXIT_CODE)"
    echo "============================================================"
    echo "  查看日志: tail -f $LOG_DIR/sft_training.log"
    echo ""
    exit $EXIT_CODE
fi
