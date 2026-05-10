#!/bin/bash
# MATH 8-Shot 对比评测: Base vs DPO 并行
# 4卡机器，每个模型2卡

set -e
cd /cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline

BASE_MODEL="Qwen2.5-Math-7B-Instruct"
DPO_MODEL="Qwen2.5-Math-7B-DPO"
OUTPUT_DIR="eval_results"
LOG_DIR="eval_results/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================"
echo "  MATH 8-Shot: Base vs DPO 并行评测"
echo "  GPU 0,1 → Base    GPU 2,3 → DPO"
echo "============================================"
echo ""

# 验证 DPO 模型存在
if [ ! -d "$DPO_MODEL" ] || [ ! -f "$DPO_MODEL/config.json" ]; then
    echo "DPO 模型不存在，先合并..."
    python3 merge_dpo_lora.py
fi

# 并行评测
CUDA_VISIBLE_DEVICES=0,1 python3 eval_math_fewshot.py \
    --model_path "$BASE_MODEL" \
    --output_path "${OUTPUT_DIR}/math_fewshot_base.jsonl" \
    --tp 2 \
    > "${LOG_DIR}/math_base.log" 2>&1 &
PID_BASE=$!
echo "[GPU 0,1] Base PID=$PID_BASE"

CUDA_VISIBLE_DEVICES=2,3 python3 eval_math_fewshot.py \
    --model_path "$DPO_MODEL" \
    --output_path "${OUTPUT_DIR}/math_fewshot_dpo.jsonl" \
    --tp 2 \
    > "${LOG_DIR}/math_dpo.log" 2>&1 &
PID_DPO=$!
echo "[GPU 2,3] DPO  PID=$PID_DPO"

echo ""
echo "日志:"
echo "  tail -f ${LOG_DIR}/math_base.log"
echo "  tail -f ${LOG_DIR}/math_dpo.log"
echo ""

# 等待
wait $PID_BASE && echo "[✓] Base 完成" || echo "[✗] Base 失败, 查看 ${LOG_DIR}/math_base.log"
wait $PID_DPO && echo "[✓] DPO  完成" || echo "[✗] DPO  失败, 查看 ${LOG_DIR}/math_dpo.log"

# 对比
echo ""
echo "============================================"
echo "  MATH 8-Shot 评测结果"
echo "============================================"
python3 -c "
import json, os

for name, fname in [('Base (Instruct)', 'math_fewshot_base_summary.json'), ('DPO', 'math_fewshot_dpo_summary.json')]:
    path = os.path.join('${OUTPUT_DIR}', fname)
    if os.path.exists(path):
        d = json.load(open(path))
        print('  %-25s %6.2f%%  (%d/%d)' % (name, d['accuracy'], d['correct'], d['total']))
    else:
        print('  %-25s 未找到' % name)

# 读两个 summary 对比
paths = ['${OUTPUT_DIR}/math_fewshot_base_summary.json', '${OUTPUT_DIR}/math_fewshot_dpo_summary.json']
summaries = []
for p in paths:
    if os.path.exists(p):
        summaries.append(json.load(open(p)))

if len(summaries) == 2:
    delta = summaries[1]['accuracy'] - summaries[0]['accuracy']
    print('  %-25s %+6.2f%%' % ('DPO vs Base', delta))
    print()
    print('  按难度对比:')
    for lv in sorted(summaries[0].get('level_stats', {}).keys()):
        base_s = summaries[0]['level_stats'].get(lv, {})
        dpo_s = summaries[1]['level_stats'].get(lv, {})
        base_acc = base_s['correct'] / base_s['total'] * 100 if base_s.get('total') else 0
        dpo_acc = dpo_s['correct'] / dpo_s['total'] * 100 if dpo_s.get('total') else 0
        d = dpo_acc - base_acc
        print('    %-10s  Base=%5.1f%%  DPO=%5.1f%%  (%+.1f%%)' % (lv, base_acc, dpo_acc, d))
"
echo "============================================"
