#!/bin/bash
# ==============================================================================
# 完整 Pipeline: 生成推理数据 + 合并 + 注册 + 训练
#
# Step 1: 启动 R1 vLLM (8卡)
# Step 2: 给 MedQA 10178 条生成推理
# Step 3: 合并所有数据
# Step 4: 训练
# ==============================================================================

cd /cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline

# Step 1: 启动 R1 (如果已经启动了可以跳过)
# bash /cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/folder/start_vllm_8gpu.sh

# Step 2: 生成推理 (~15分钟)
echo "=== Step 2: Generating reasoning for MedQA ==="
python generate_reasoning_medqa.py \
  --input  train-00000-of-00001.parquet \
  --output medqa_reasoning_sharegpt.json \
  --num_instances 8 \
  --base_port 8000 \
  --max_workers 64

# Step 3: 合并 MedQA推理数据 + MCQ v2数据 为最终训练集
# 注意: folder/train.json 和 pipeline/(1).parquet 是同一批182822条, 只用一份避免重复
echo "=== Step 3: Merging all data ==="
python -c "
import json
# MedQA 推理数据 (R1生成, 10178条)
d1 = json.load(open('medqa_reasoning_sharegpt.json'))
# MCQ v2 数据 (folder下的, 有exp + 字母映射, 182822条)
d2 = json.load(open('/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/folder/train_sharegpt_v2.json'))

merged = d1 + d2
with open('final_reasoning_train.json', 'w') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f'MedQA (R1 reasoning): {len(d1)}')
print(f'MCQ (exp + letter mapping): {len(d2)}')
print(f'Total merged: {len(merged)}')
"

echo ""
echo "=== Done! ==="
echo "Final dataset: /cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/final_reasoning_train.json"
echo ""
echo "Next: register in dataset_info.json and run training"
