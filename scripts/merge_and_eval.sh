#!/usr/bin/env bash
# ============================================================
# FSDP Checkpoint 合并 + MATH Zero-Shot 对比评测
#
# 功能：
#   1. 将 veRL FSDP 分片 checkpoint 合并为 HuggingFace 格式
#   2. GRPO 模型 vs Base 模型并行推理，各占一组 GPU
#   3. 输出对比结果
#
# 用法:
#   bash merge_and_eval.sh                    # 合并最新step + 对比评测
#   GRPO_STEP=2400 bash merge_and_eval.sh     # 指定 step
#   LIMIT=50 bash merge_and_eval.sh           # 快速验证（只跑50题）
#   SKIP_MERGE=1 bash merge_and_eval.sh       # 跳过合并（已合并过）
#   SKIP_EVAL=1 bash merge_and_eval.sh        # 只合并不评测
#   NO_BASE=1 bash merge_and_eval.sh          # 不评测 base（只跑GRPO）
#
# 环境变量:
#   GRPO_RUN_NAME    checkpoint 运行名称
#   GRPO_STEP        要合并的 step（留空则用 latest）
#   CUDA_GRPO        GRPO 评测 GPU（默认 0,1,2,3）
#   CUDA_BASE        Base 评测 GPU（默认 4,5,6,7）
#   TP               tensor parallel（默认 4）
#   GPU_UTIL         显存利用率（默认 0.85）
#   LIMIT            评测题数（留空=全量）
#   SKIP_MERGE       =1 跳过合并
#   SKIP_EVAL        =1 跳过评测
#   NO_BASE          =1 不评测 base model
#   BASE_MODEL       base model 路径
# ============================================================

set -euo pipefail

# ===== 基础路径 =====
PIPELINE_DIR="/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline"
VERL_DIR="/cfs/cfs-esygraib/belvathliu/cv3/verl"

# ===== 配置参数 =====
GRPO_RUN_NAME="${GRPO_RUN_NAME:-qwen25_math_grpo_warmstart_v4_20260503_rlvr_balanced}"
GRPO_STEP="${GRPO_STEP:-}"
CUDA_GRPO="${CUDA_GRPO:-0,1,2,3}"
CUDA_BASE="${CUDA_BASE:-4,5,6,7}"
TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.85}"
LIMIT="${LIMIT:-}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
NO_BASE="${NO_BASE:-0}"
BASE_MODEL="${BASE_MODEL:-${PIPELINE_DIR}/Qwen2.5-Math-7B-Instruct}"

# ===== 解析 checkpoint 路径 =====
CKPT_ROOT="${PIPELINE_DIR}/checkpoints/${GRPO_RUN_NAME}"

if [[ -z "$GRPO_STEP" ]]; then
    LATEST_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"
    if [[ -f "$LATEST_FILE" ]]; then
        GRPO_STEP="$(tr -d '[:space:]' < "$LATEST_FILE")"
        echo "[信息] 自动检测最新 step: ${GRPO_STEP}"
    else
        echo "[错误] 未指定 GRPO_STEP 且找不到 ${LATEST_FILE}"
        exit 1
    fi
fi

CKPT_DIR="${CKPT_ROOT}/global_step_${GRPO_STEP}"
ACTOR_DIR="${CKPT_DIR}/actor"
MERGED_DIR="${CKPT_DIR}/actor_merged_hf"

# ===== 验证 checkpoint 存在 =====
if [[ ! -d "$ACTOR_DIR" ]]; then
    echo "[错误] Checkpoint 目录不存在: ${ACTOR_DIR}"
    echo ""
    echo "可用的 steps:"
    ls -d "${CKPT_ROOT}"/global_step_* 2>/dev/null | sed 's/.*global_step_/  /' || echo "  (无)"
    exit 1
fi

echo "============================================================"
echo "  FSDP Merge + MATH Zero-Shot 对比评测"
echo "============================================================"
echo "运行名称:       ${GRPO_RUN_NAME}"
echo "Step:           ${GRPO_STEP}"
echo "Checkpoint:     ${ACTOR_DIR}"
echo "合并输出:       ${MERGED_DIR}"
echo "GRPO 模型GPU:   ${CUDA_GRPO} (TP=${TP})"
if [[ "$NO_BASE" != "1" ]]; then
    echo "BASE 模型:      ${BASE_MODEL}"
    echo "BASE 模型GPU:   ${CUDA_BASE} (TP=${TP})"
fi
echo "GPU利用率:      ${GPU_UTIL}"
if [[ -n "$LIMIT" ]]; then
    echo "评测题数:       ${LIMIT}"
else
    echo "评测题数:       全量"
fi
echo "============================================================"
echo ""

# =============================================================
# Step 1: 合并 FSDP 分片
# =============================================================
if [[ "$SKIP_MERGE" != "1" ]]; then
    if [[ -d "$MERGED_DIR" && -f "$MERGED_DIR/config.json" ]]; then
        echo "[合并] 已存在合并后的模型，跳过: ${MERGED_DIR}"
    else
        echo "[合并] 开始合并 FSDP 分片..."
        echo "  输入: ${ACTOR_DIR}"
        echo "  输出: ${MERGED_DIR}"
        echo ""

        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$MERGED_DIR"

        if [[ ! -f "$MERGED_DIR/config.json" ]]; then
            echo "[错误] 合并失败：${MERGED_DIR}/config.json 不存在"
            exit 1
        fi

        echo ""
        echo "[合并] 完成！"
        ls -lh "$MERGED_DIR"/*.safetensors 2>/dev/null || ls -lh "$MERGED_DIR"/*.bin 2>/dev/null || true
        echo ""
    fi
else
    echo "[合并] 跳过（SKIP_MERGE=1）"
    if [[ ! -d "$MERGED_DIR" || ! -f "$MERGED_DIR/config.json" ]]; then
        echo "[错误] SKIP_MERGE=1 但合并模型不存在: ${MERGED_DIR}"
        exit 1
    fi
fi

# =============================================================
# Step 2: MATH Zero-Shot 并行评测
# =============================================================
if [[ "$SKIP_EVAL" != "1" ]]; then
    echo ""
    echo "[评测] 开始 MATH Zero-Shot 评测..."

    RESULT_DIR="${PIPELINE_DIR}/eval_results_grpo/${GRPO_RUN_NAME}"
    mkdir -p "$RESULT_DIR"

    GRPO_JSON="${RESULT_DIR}/math_zeroshot_step${GRPO_STEP}.json"
    GRPO_LOG="${RESULT_DIR}/math_zeroshot_step${GRPO_STEP}.log"
    BASE_JSON="${RESULT_DIR}/math_zeroshot_base.json"
    BASE_LOG="${RESULT_DIR}/math_zeroshot_base.log"

    LIMIT_ARGS=()
    if [[ -n "$LIMIT" ]]; then
        LIMIT_ARGS=(--limit "$LIMIT")
    fi

    # --- 启动 GRPO 评测（后台）---
    echo "  [GRPO] GPU=${CUDA_GRPO}  模型=${MERGED_DIR}"
    CUDA_VISIBLE_DEVICES="$CUDA_GRPO" python3 "${PIPELINE_DIR}/eval_math.py" \
        --model "$MERGED_DIR" \
        --dataset math \
        --tp "$TP" \
        --gpu_util "$GPU_UTIL" \
        --output "$GRPO_JSON" \
        "${LIMIT_ARGS[@]}" \
        > "$GRPO_LOG" 2>&1 &
    PID_GRPO=$!
    echo "  [GRPO] 已启动 PID=${PID_GRPO}"

    # --- 启动 BASE 评测（后台，并行）---
    PID_BASE=""
    if [[ "$NO_BASE" != "1" ]]; then
        if [[ -d "$BASE_MODEL" && -f "$BASE_MODEL/config.json" ]]; then
            echo "  [BASE] GPU=${CUDA_BASE}  模型=${BASE_MODEL}"
            CUDA_VISIBLE_DEVICES="$CUDA_BASE" python3 "${PIPELINE_DIR}/eval_math.py" \
                --model "$BASE_MODEL" \
                --dataset math \
                --tp "$TP" \
                --gpu_util "$GPU_UTIL" \
                --output "$BASE_JSON" \
                "${LIMIT_ARGS[@]}" \
                > "$BASE_LOG" 2>&1 &
            PID_BASE=$!
            echo "  [BASE] 已启动 PID=${PID_BASE}"
        else
            echo "  [BASE] 警告: 模型目录不存在或缺少config.json: ${BASE_MODEL}"
        fi
    fi

    # --- 等待完成 ---
    echo ""
    echo "[评测] 推理进行中... (可用 tail -f ${GRPO_LOG} 查看进度)"
    echo ""

    FAILED=0
    if wait "$PID_GRPO"; then
        echo "  [✓] GRPO 评测完成"
    else
        echo "  [✗] GRPO 评测失败，查看: ${GRPO_LOG}"
        FAILED=$((FAILED + 1))
    fi

    if [[ -n "$PID_BASE" ]]; then
        if wait "$PID_BASE"; then
            echo "  [✓] BASE 评测完成"
        else
            echo "  [✗] BASE 评测失败，查看: ${BASE_LOG}"
            FAILED=$((FAILED + 1))
        fi
    fi

    # =============================================================
    # Step 3: 结果汇总
    # =============================================================
    echo ""
    echo "============================================================"
    echo "  结果汇总 (Step ${GRPO_STEP})"
    echo "============================================================"

    # 直接用 shell 变量传给 python（不依赖 export/environ）
    python3 - "$GRPO_JSON" "$BASE_JSON" "$GRPO_STEP" "$NO_BASE" <<'PY'
import json, os, sys

grpo_json = sys.argv[1]
base_json = sys.argv[2]
step = sys.argv[3]
no_base = sys.argv[4]

results = {}

# GRPO 结果
if os.path.exists(grpo_json):
    with open(grpo_json, "r") as f:
        data = json.load(f)
    res = data.get("math", data)
    acc = res.get("accuracy", 0.0)
    correct = res.get("correct", 0)
    total = res.get("total", 0)
    results["GRPO"] = acc
    print(f"  GRPO (step {step}):  {acc:>6.2f}%  ({correct}/{total})")
else:
    print(f"  GRPO: 结果文件不存在 {grpo_json}")

# BASE 结果
if no_base != "1" and os.path.exists(base_json):
    with open(base_json, "r") as f:
        data = json.load(f)
    res = data.get("math", data)
    acc = res.get("accuracy", 0.0)
    correct = res.get("correct", 0)
    total = res.get("total", 0)
    results["BASE"] = acc
    print(f"  BASE (Instruct):   {acc:>6.2f}%  ({correct}/{total})")

# 对比
if "GRPO" in results and "BASE" in results:
    delta = results["GRPO"] - results["BASE"]
    print(f"  ────────────────────────────")
    print(f"  GRPO - BASE:       {delta:+.2f}%")
PY

    echo "============================================================"
    echo ""
    echo "日志查看:"
    echo "  tail -f ${GRPO_LOG}"
    [[ -n "$PID_BASE" ]] && echo "  tail -f ${BASE_LOG}"
    echo ""
    echo "失败任务数: ${FAILED}"

else
    echo "[评测] 跳过（SKIP_EVAL=1）"
fi

echo ""
echo "[完成] 合并模型路径: ${MERGED_DIR}"
