#!/usr/bin/env bash
# 对比 base model 和 GRPO merge 模型的 MATH zero-shot 成绩
#
# 用法:
#   bash run_math_zeroshot_compare.sh
#
# 可选环境变量:
#   BASE_MODEL=/path/to/base_model
#   GRPO_MODEL=/path/to/grpo_model
#   GRPO_RUN_NAME=qwen25_math_grpo_run2_20260501
#   GRPO_STEP=1750
#   CUDA_GRPO=0,1,2,3
#   CUDA_BASE=4,5,6,7
#   TP_GRPO=4
#   TP_BASE=4
#   GPU_UTIL=0.85
#   LIMIT=100

set -euo pipefail

PIPELINE_DIR="/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline"
BASE_MODEL="${BASE_MODEL:-${PIPELINE_DIR}/Qwen2.5-Math-7B-Instruct}"
GRPO_RUN_NAME="${GRPO_RUN_NAME:-qwen25_math_grpo_run2_20260501}"
GRPO_STEP="${GRPO_STEP:-}"
CUDA_GRPO="${CUDA_GRPO:-0,1,2,3}"
CUDA_BASE="${CUDA_BASE:-4,5,6,7}"
TP_GRPO="${TP_GRPO:-4}"
TP_BASE="${TP_BASE:-4}"
GPU_UTIL="${GPU_UTIL:-0.85}"
LIMIT="${LIMIT:-}"

resolve_default_grpo_model() {
  if [[ -n "${GRPO_MODEL:-}" ]]; then
    echo "$GRPO_MODEL"
    return 0
  fi

  local run_dir latest_step_file resolved_step
  run_dir="${PIPELINE_DIR}/checkpoints/${GRPO_RUN_NAME}"
  latest_step_file="${run_dir}/latest_checkpointed_iteration.txt"

  if [[ -n "$GRPO_STEP" ]]; then
    resolved_step="$GRPO_STEP"
  elif [[ -f "$latest_step_file" ]]; then
    resolved_step="$(tr -d '[:space:]' < "$latest_step_file")"
  else
    echo "${PIPELINE_DIR}/Qwen2.5-Math-7B-GRPO-1750"
    return 0
  fi

  echo "${run_dir}/global_step_${resolved_step}/actor_merged_hf"
}

GRPO_MODEL="$(resolve_default_grpo_model)"

count_csv_items() {
  local csv="$1"
  awk -F',' '{print NF}' <<< "$csv"
}

validate_model_dir() {
  local name="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "$name 模型目录不存在: $path"
    exit 1
  fi
  if [[ ! -f "$path/config.json" ]]; then
    echo "$name 缺少 config.json: $path"
    exit 1
  fi
}

validate_tp() {
  local name="$1"
  local devices="$2"
  local tp="$3"
  local gpu_count
  gpu_count="$(count_csv_items "$devices")"

  case "$tp" in
    1|2|4|7|14|28) ;;
    *)
      echo "$name: 非法 TP=$tp。Qwen2.5-Math-7B 只支持 TP in {1,2,4,7,14,28}"
      exit 1
      ;;
  esac

  if [[ "$tp" -gt "$gpu_count" ]]; then
    echo "$name: TP=$tp 大于可见 GPU 数量=$gpu_count。CUDA_VISIBLE_DEVICES=$devices"
    exit 1
  fi
}

validate_model_dir "BASE" "$BASE_MODEL"
validate_model_dir "GRPO" "$GRPO_MODEL"
validate_tp "BASE" "$CUDA_BASE" "$TP_BASE"
validate_tp "GRPO" "$CUDA_GRPO" "$TP_GRPO"

RESULT_ROOT="${PIPELINE_DIR}/eval_results_compare/math_zeroshot_base_vs_grpo"
LOG_DIR="${RESULT_ROOT}/logs"
mkdir -p "$RESULT_ROOT" "$LOG_DIR"

GRPO_JSON="${RESULT_ROOT}/math_zeroshot_grpo.json"
BASE_JSON="${RESULT_ROOT}/math_zeroshot_base.json"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "============================================================"
echo "  MATH Zero-Shot 对比评测"
echo "============================================================"
echo "GRPO 模型:      $GRPO_MODEL"
echo "BASE 模型:      $BASE_MODEL"
echo "GRPO GPUs:      $CUDA_GRPO (TP=$TP_GRPO)"
echo "BASE GPUs:      $CUDA_BASE (TP=$TP_BASE)"
echo "GPU_UTIL:       $GPU_UTIL"
if [[ -n "$LIMIT" ]]; then
  echo "LIMIT:          $LIMIT"
else
  echo "LIMIT:          full"
fi
echo "结果目录:       $RESULT_ROOT"
echo "日志目录:       $LOG_DIR"
echo "============================================================"
echo ""

CUDA_VISIBLE_DEVICES="$CUDA_GRPO" python3 "${PIPELINE_DIR}/eval_math.py" \
  --model "$GRPO_MODEL" \
  --dataset math \
  --tp "$TP_GRPO" \
  --gpu_util "$GPU_UTIL" \
  --output "$GRPO_JSON" \
  "${LIMIT_ARGS[@]}" \
  > "${LOG_DIR}/grpo.log" 2>&1 &
PID_GRPO=$!
echo "[GRPO] 已启动 PID=$PID_GRPO"

CUDA_VISIBLE_DEVICES="$CUDA_BASE" python3 "${PIPELINE_DIR}/eval_math.py" \
  --model "$BASE_MODEL" \
  --dataset math \
  --tp "$TP_BASE" \
  --gpu_util "$GPU_UTIL" \
  --output "$BASE_JSON" \
  "${LIMIT_ARGS[@]}" \
  > "${LOG_DIR}/base.log" 2>&1 &
PID_BASE=$!
echo "[BASE] 已启动 PID=$PID_BASE"
echo ""

FAILED=0
if wait "$PID_GRPO"; then
  echo "[✓] GRPO 完成: $GRPO_JSON"
else
  echo "[✗] GRPO 失败，查看 ${LOG_DIR}/grpo.log"
  FAILED=$((FAILED + 1))
fi

if wait "$PID_BASE"; then
  echo "[✓] BASE 完成: $BASE_JSON"
else
  echo "[✗] BASE 失败，查看 ${LOG_DIR}/base.log"
  FAILED=$((FAILED + 1))
fi

echo ""
echo "============================================================"
echo "  结果汇总"
echo "============================================================"
python3 - <<PY
import json, os

files = {
    "GRPO": "${GRPO_JSON}",
    "BASE": "${BASE_JSON}",
}

results = {}
for name, path in files.items():
    if not os.path.exists(path):
        print(f"{name:<8} 未找到结果文件: {path}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = data.get("math", {})
    results[name] = res
    print(f"{name:<8} {res.get('accuracy', 0.0):>6.2f}%  ({res.get('correct', 0)}/{res.get('total', 0)})")

if "GRPO" in results and "BASE" in results:
    delta = results["GRPO"].get("accuracy", 0.0) - results["BASE"].get("accuracy", 0.0)
    print("--------")
    print(f"GRPO-BASE {delta:+.2f}%")
PY
echo "============================================================"
echo "失败任务数:    $FAILED"
echo "日志查看:"
echo "  tail -f ${LOG_DIR}/grpo.log"
echo "  tail -f ${LOG_DIR}/base.log"
echo "============================================================"
