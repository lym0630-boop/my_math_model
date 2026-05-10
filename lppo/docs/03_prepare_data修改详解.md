# prepare_math_rlvr_data.py 修改详解

## 文件位置
`pipline/prepare_math_rlvr_data.py`

## 修改量
**3 行代码**：1 个 import + 1 个函数定义 + 1 行字段添加

---

## 修改内容

### 1. 新增 import（文件顶部）
```python
import hashlib  # ← 新增
```

### 2. 新增函数（在 `SYSTEM_PROMPT_COT` 之后）
```python
def make_sample_id(question: str, ground_truth: str) -> str:
    """
    基于题目+答案生成唯一且稳定的 sample_id
    与 lppo/lp_init.py 中的 compute_sample_id() 完全一致。
    """
    key = question.strip() + "||" + ground_truth.strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]
```

### 3. 在 `extra_info` dict 中加一行（L355）
```python
"extra_info": {
    "sample_id": make_sample_id(c['question'], c['ground_truth']),  # ← 新增
    "category": c['category'],
    "student_accuracy": c.get('student_accuracy'),
    "teacher_status": c.get('teacher_status'),
    "difficulty_bucket": difficulty_bucket(...),
},
```

---

## 设计决策

### Q: 为什么用 `question + ground_truth` 做 hash？

**A**: 
- 同一个问题文本可能因为 GT 不同而对应不同的"题目"
- 例如："求 x^2 = 4 的解" → GT 可能是 "2" 也可能是 "-2,2"
- 用 question 单独做 hash 可能产生冲突

### Q: 为什么用 SHA1 而不是 MD5？

**A**:
- SHA1 碰撞概率更低（160 bit vs 128 bit）
- 取前 16 位 hex = 64 bit 空间
- 对 120K 规模的题库，碰撞概率 ≈ (120000)^2 / 2^64 ≈ 8×10^-10
- 完全安全

### Q: 为什么不直接用行号/序号？

**A**:
- 行号会因为重新筛选/排序而变化
- sample_id 需要**跨 cycle 稳定**——同一道题在 cycle 1 和 cycle 5 应该有相同的 ID
- 这样 LP state 才能正确追踪同一道题的学习进度

### Q: sample_id 在训练中是怎么被使用的？

**A**: 
```
parquet (sample_id in extra_info)
    → verl dataloader 加载
    → batch.non_tensor_batch["extra_info"] (list of dicts)
    → ray_trainer Hook B 提取 sample_id
    → LP State Manager 按 sample_id 分组 + 更新
```

---

## 验证方法

```python
# 验证 sample_id 在 parquet 中正确存储
import pandas as pd
df = pd.read_parquet("data/parquet_math_rlvr_v4_balanced/train.parquet")
sample = df.iloc[0]
print(sample['extra_info']['sample_id'])  # 应该是 16 位 hex 字符串
```

---

## 对训练的影响

- **无性能影响**：hashlib.sha1 是 O(n) 且只在数据准备时运行一次
- **不影响现有逻辑**：extra_info 中的其他字段完全不变
- **向后兼容**：如果 ray_trainer 没有开启 LPPO，sample_id 只是一个被忽略的字段
