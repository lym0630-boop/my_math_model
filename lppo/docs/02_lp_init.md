# lp_init.py 详解

## 文件定位
`pipline/lppo/lp_init.py` — LP 状态初始化模块，从历史数据生成 P0。

---

## 为什么需要这个文件

### 问题：冷启动偏差
如果 LP state 从全零开始：
- Step 1: 某题 pass_rate=0.3 → p 从 0 变为 0.3，lp=+0.3（巨大正进步）
- Step 2: 同一题 pass_rate=0.3 → p 从 0.3 变为 0.3，lp=0（正常）

第一个 step 的 "虚假 LP" 会导致所有题在第一次出现时获得极高权重，之后骤降。

### 解决：用历史数据初始化
student_responses_*.jsonl 已经记录了每道题在 DPO-v2 模型上的 pass rate，
直接用它初始化 p，就跳过了冷启动阶段。

---

## 核心函数

### `compute_sample_id(question)`

```python
def compute_sample_id(question: str) -> str:
    key = question[:200].strip()
    return hashlib.md5(key.encode()).hexdigest()[:12]
```

**设计决策**：
- 用 MD5 前 12 位（hex）= 48 bit = 2^48 种可能，15000 题碰撞概率 < 10^-6
- 取前 200 字符：避免尾部差异（如多余空格）导致不同 ID
- 与 `prepare_math_rlvr_data.py` 中使用同一函数，确保一致性

**面试解释**：
> sample_id 的设计需要满足两个要求：一是稳定（同一题始终得到同一 ID），二是唯一（不同题不碰撞）。MD5 hash 的前 12 位 hex 满足这两点，且长度适中便于日志阅读。

---

### `init_lp_state_from_student_responses()`

```python
def init_lp_state_from_student_responses(student_responses_path, lp_manager, output_path):
    # 逐行读取 student_responses
    for line in f:
        d = json.loads(line)
        sample_id = compute_sample_id(d['question'])
        pass_rate = d['num_correct'] / max(d['num_total'], 1)
        
        # 直接设置初始状态（不走 EMA update）
        lp_manager.states[sample_id] = {
            'p': pass_rate,   # 历史 pass rate 作为 P0
            'lp': 0.0,        # 初始 LP = 0（无变化信息）
            'n_updates': 1,   # 标记已初始化
        }
```

**为什么不用 `update()` 方法？**
- `update()` 会走 EMA 逻辑，第一次调用时 p_old=0，会产生 lp = pass_rate 的虚假进步
- 直接赋值跳过 EMA，保证 lp=0，训练开始后第一次真正的 update 才产生有意义的 lp

---

## 使用方式

### 独立运行（Stage 1.5）
```bash
python3 -m lppo.lp_init \
    --student_responses sft_data/student_responses_120k_expanded_v4.jsonl \
    --output lppo/lp_state_init.json
```

### 在 ray_trainer.py 中自动加载
```python
# __init__ 中
from lppo.lp_init import init_lp_state_from_student_responses
lp_mgr = LPStateManager(beta=0.8, ...)
init_lp_state_from_student_responses(student_responses_path, lp_mgr)
```

---

## 输出文件格式

`lp_state_init.json` 示例：
```json
{
  "config": {"beta": 0.8, "w_min": 0.25, "w_max": 2.0, ...},
  "states": {
    "a3f2c1d9e8b7": {"p": 0.375, "lp": 0.0, "n_updates": 1},
    "c7e4a2b1f8d9": {"p": 0.0,   "lp": 0.0, "n_updates": 1},
    ...
  },
  "summary": {
    "n_problems": 15000,
    "avg_p": 0.287,
    "hard_zero_count": 3200,
    "sweet_spot_count": 5800,
    "mastered_count": 1200
  }
}
```

---

## 面试追问预判

### Q: student_responses 中的 pass rate 是用哪个模型采样的？
**A**: 是 DPO-v2 模型（即 WarmStart 之前的模型）。WarmStart 之后模型能力会有变化，但 DPO-v2 的 pass rate 仍然是很好的初始估计。训练开始后 EMA 会快速修正。

### Q: 如果训练从 checkpoint 恢复，P0 还需要重新初始化吗？
**A**: 不需要。恢复时从 checkpoint 同保存的 LP state JSON 加载，那里已经有训练中更新过的 p 值了。P0 初始化只在第一次启动训练时做一次。
