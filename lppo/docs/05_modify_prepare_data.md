# prepare_math_rlvr_data.py 修改详解

## 修改目标
在 `extra_info` 中添加 `sample_id` 字段，为 LPPO 提供题目唯一标识。

---

## 修改位置
文件：`pipline/prepare_math_rlvr_data.py`
行号：L355-360（`extra_info` 字典构造处）

---

## 修改前 (原始代码)

```python
# L341-361
records = []
for c in candidates:
    records.append({
        "data_source": "math",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT_COT},
            {"role": "user", "content": c['question']},
            {"role": "assistant", "content": ""},
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": c['ground_truth'],
        },
        "extra_info": {
            "category": c['category'],
            "student_accuracy": c.get('student_accuracy'),
            "teacher_status": c.get('teacher_status'),
            "difficulty_bucket": difficulty_bucket(c.get('student_accuracy'), c.get('teacher_status')),
        },
    })
```

---

## 修改后 (添加 sample_id)

```python
import hashlib  # ← 在文件顶部添加（实际已有 import re, random 等）

def compute_sample_id(question: str) -> str:
    """根据题目文本生成稳定的 sample_id（与 lppo/lp_init.py 一致）"""
    key = question[:200].strip()
    return hashlib.md5(key.encode()).hexdigest()[:12]

# L341-361 修改版
records = []
for c in candidates:
    sample_id = compute_sample_id(c['question'])  # ← 新增
    records.append({
        "data_source": "math",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT_COT},
            {"role": "user", "content": c['question']},
            {"role": "assistant", "content": ""},
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": c['ground_truth'],
        },
        "extra_info": {
            "sample_id": sample_id,                 # ← 新增：LPPO 用的题目 ID
            "category": c['category'],
            "student_accuracy": c.get('student_accuracy'),
            "teacher_status": c.get('teacher_status'),
            "difficulty_bucket": difficulty_bucket(c.get('student_accuracy'), c.get('teacher_status')),
        },
    })
```

---

## 改动说明

| 项目 | 说明 |
|------|------|
| 改动量 | 新增约 8 行（函数 + 调用） |
| 风险 | 极低。只在 dict 中加了一个字段，不影响任何下游逻辑 |
| 兼容性 | 完全向后兼容。旧代码不读 sample_id 就无影响 |
| 依赖 | 只用 `hashlib`（标准库） |

---

## 为什么放在 extra_info 而不是顶层

verl 框架对顶层字段有严格要求（data_source, prompt, ability, reward_model），
额外字段放在 extra_info 中是 verl 的标准扩展方式：
- `extra_info` 是 verl 官方支持的"用户自定义元数据"字段
- 会被传递到 `batch.non_tensor_batch` 中供后续使用
- ray_trainer.py 中可以通过 `batch.non_tensor_batch["extra_info"]` 访问

---

## 验证方法

```bash
# 重新生成数据
python3 prepare_math_rlvr_data.py --output_dir /tmp/test_data

# 验证 sample_id 存在且唯一
python3 -c "
import pandas as pd
df = pd.read_parquet('/tmp/test_data/train.parquet')
ids = [row['extra_info']['sample_id'] for _, row in df.iterrows()]
print(f'总行数: {len(ids)}')
print(f'唯一 ID 数: {len(set(ids))}')
print(f'示例 ID: {ids[:5]}')
assert len(ids) == len(set(ids)), '有重复 ID！'
print('✅ 验证通过：所有 sample_id 唯一')
"
```

---

## sample_id 在后续流程中的流转

```
prepare_math_rlvr_data.py
  └── 写入 parquet: extra_info.sample_id = "a3f2c1d9e8b7"
      └── verl DataLoader 加载
          └── ray_trainer.py 中 batch.non_tensor_batch 可访问
              └── Hook B: 用 sample_id 查询 LP state
              └── Hook D: 用 sample_id 记录学习进度
```

---

## 面试话术

> 这个修改很小但很关键——它是 LPPO 的"桥梁"。sample_id 让训练过程中的每个 batch 样本都能追溯回具体的题目，从而查询和更新 LP state。选用 MD5 hash 而非顺序编号是因为 hash 是确定性的：同一道题无论在哪个 cycle、哪次数据重建中，都会得到相同的 ID，确保 LP 状态的连续性。
