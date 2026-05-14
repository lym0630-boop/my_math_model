# Math Training Pipeline

基于 **Qwen2.5-Math-7B** 的数学推理能力增强全流程，涵盖 CPT 领域续训、SFT 监督微调、DPO 偏好优化、Prefix-Guided Warm-Start 以及 LPPO 强化学习。

> **注意**：本仓库仅包含代码，不含模型权重、数据集、Checkpoint 及训练日志。克隆后需根据自身环境修改脚本中的绝对路径。

---
**评测结果**
| Model      | Math Eval     |
|------------|---------------|
| Base Model | 83.0%         |
| LPPO v2    | 84.2% (+1.2%) |


## 项目结构

```
.
├── pipelines/          # DPO 数据构造核心流程（选题、采样、组装）
├── training/           # SFT / DPO 数据转换与 LoRA 合并工具
├── evaluation/         # GSM8K / MATH 评测脚本
├── rlvr/               # LPPO 训练数据准备、Reward 函数、Prefix-Guided Warm-Start
├── lppo/               # LPPO 核心模块（Learning Progress 状态管理、动态权重、前缀引导 Rollout）
├── scripts/            # 各阶段 Shell 启动脚本
├── modules/
│   ├── data_cleaner/   # 通用数学语料二次清洗 Pipeline（段切分 → 规则清洗 → Classifier 路由）
│   ├── openwebmath/       # OpenWebMath-4plus CPT 数据处理
│   └── eval_tasks/     # lm-evaluation-harness 本地任务配置
├── utils/              # 零散维护工具
└── requirements.txt
```

---

## 训练流程总览

```
OpenWebMath-4plus 原始网页数学语料
     │
     ▼
[1] CPT 数据清洗（规则粗切 + 大模型筛选补充）
     │  modules/data_cleaner/    — 段切分 → 轻规则 → Classifier → LLM 补全 → 校验
     │  modules/openwebmath/clean_openwebmath.py  — 快速预过滤
     ▼
[2] CPT 领域续训（全参数，DeepSpeed ZeRO-2）
     │  modules/openwebmath/run_cpt.sh
     │  modules/openwebmath/qwen25_math_cpt.yaml
     ▼
[3] SFT 数据构造 & CoT 数据清洗
     │  modules/openwebmath/convert_qa_to_sft.py  — QA → SFT 格式
     │  modules/openwebmath/clean_sft_data.py     — 答案格式统一 + 噪声过滤
     ▼
[4] SFT 监督微调（LoRA）
     │  scripts/run_sft.sh
     │  training/qwen25_math_sft.yaml
     ▼
[5] DPO 数据构造（Student-Teacher 框架）
     │  pipelines/select_dpo_questions.py      — 学科分层选题
     │  pipelines/dpo_final_pipeline.py        — Student 8×采样 + Teacher 2×rescue
     │  pipelines/dpo_iterative_pipeline.py    — 多轮迭代补采
     ▼
[6] DPO 偏好优化训练（LoRA）
     │  scripts/run_dpo_final.sh
     ▼
[7] Prefix-Guided Warm-Start SFT
     │  rlvr/prefix_guided_warmstart.py        — 对 hard bucket 做前缀引导续写
     ▼
[8] LPPO 强化学习（基于 verl + GRPO）
     │  rlvr/prepare_math_rlvr_data.py
     │  rlvr/reward_math_rlvr.py
     │  scripts/run_math_grpo.sh
     ▼
[9] LoRA 合并 + 评测
     │  training/merge_lora_weights.py
     │  evaluation/eval_math.py
```

---


#### CoT 数据构造

- `convert_qa_to_sft.py`：从 swallowmath_qa 原始文本提取 QA 对，转换为 sharegpt 格式
- `clean_sft_data.py`：统一答案格式（补 `\boxed{}`）、截断 *Note:* 废话、过滤浮点精度垃圾

#### SFT 训练

基于 CPT checkpoint 做 LoRA SFT（rank=64, alpha=128）：

```bash
bash scripts/run_sft.sh
```

---

###  DPO 数据构造（`pipelines/`）

#### 选题策略 (`select_dpo_questions.py`)

从候选池筛选适合 DPO 的高质量题目：

1. **答案可验证性**：只保留 GT 来源为 `code_output` 且为纯数值/分数的题
2. **学科多样性**：按类别分层采样（代数 25%、几何 12%、概率统计 10% ...）
3. **排除非数学**：去掉物理和 CS/算法题
4. **数值类型控制**：大数(>10000)不超过 5%
5. **去重**：前120字符去重

#### Student-Teacher 数据构造 (`dpo_final_pipeline.py`)


```bash
# Step 1: Student 推理（4×A100，约 3h）
python3 pipelines/dpo_final_pipeline.py --stage student

# Step 2: Teacher 推理（4×A100，约 1h）
python3 pipelines/dpo_final_pipeline.py --stage teacher

# Step 3: 组装（无需 GPU）
python3 pipelines/dpo_final_pipeline.py --stage assemble --target 8000
```

支持**多卡分片并行**：

```bash
python3 pipelines/dpo_final_pipeline.py --stage student --num_shards 2 --shard_id 0
python3 pipelines/dpo_final_pipeline.py --stage student --num_shards 2 --shard_id 1
python3 pipelines/dpo_final_pipeline.py --stage merge_student --num_shards 2
```

#### 迭代补采 (`dpo_iterative_pipeline.py`)

当单轮数据量不足 target 时，从候选池补采新题并重复 Student-Teacher 流程。

---

### Prefix-Guided Warm-Start（`rlvr/prefix_guided_warmstart.py`）

在 DPO 之后、LPPO 之前，对 **hard bucket**（Student 8次全错）的题做定向抢救：

| 阶段 | 说明 |
|------|------|
| prepare | 截取参考解前缀（短20%/中40%/长60%三档），含答案泄露检查 |
| generate | DPO 模型基于前缀续写（每档×4次采样） |
| assemble | 过滤正确且非抄袭的样本（LCS + ROUGE-L），组装 SFT 数据 |
| evaluate | Gate 评测：warm-start 后模型在 hard 题上的 solve rate ≥ 5% 才进入 LPPO |

**过滤条件**：
- 答案必须正确
- 续写 ≥ 100 字符
- 前50字符不能直接出现 GT
- 最长公共子串(LCS) ≤ 50
- ROUGE-L ≤ 0.7

```bash
python3 rlvr/prefix_guided_warmstart.py --stage prepare
python3 rlvr/prefix_guided_warmstart.py --stage generate
python3 rlvr/prefix_guided_warmstart.py --stage assemble
python3 rlvr/prefix_guided_warmstart.py --stage evaluate --model_path /path/to/warmstart_model
```

---

### LPPO 强化学习（`lppo/` + `rlvr/` + `scripts/run_math_grpo.sh`）（参考 https://arxiv.org/html/2507.06573v1 ）

基于 [verl](https://github.com/volcengine/verl) 框架的 **GRPO + LPPO**（Learning Progress Prioritized Policy Optimization）训练。

#### `lppo/` 模块

| 文件 | 功能 |
|------|------|
| `lp_state_manager.py` | LP 核心状态管理：EMA pass rate、Learning Progress 计算、sigmoid 权重映射 |
| `lp_init.py` | 从历史 Student 采样结果初始化 P₀，跳过冷启动阶段 |
| `prefix_guided_rollout.py` | 为 hard-zero 题准备带前缀的 prompt（复用 warm-start 逻辑） |
| `build_cycle_data.py` | 基于 LP state 的 Cycle 重采样（动态调整训练数据分布） |
| `build_active_pool.py` | 构建 active 训练池 |
| `refresh_pg.py` | 刷新 prefix-guided 数据 |
| `docs/` | LPPO 设计文档与实现详解 |

#### LPPO 核心思想

使用 **Learning Progress**（学习进度）动态调整每道题的采样权重：
- 从 Student 历史采样结果计算初始通过率 P₀
- 训练过程中通过 EMA 追踪每题通过率变化
- 正在「学会」的题（通过率上升）权重增大（w_max=2.0）
- 已经「学会」或「太难」的题权重减小（w_min=0.25）
- 避免训练初期的"虚假学习进度"


#### 训练配置

- basemodel：Qwen2.5-Math-7B-WarmStart（经过 CPT → SFT → DPO → Warm-Start）
- **算法**：GRPO + LPPO
- **硬件**：8×A100 
- **关键超参**：
  - rollout_n=8, response_length=1024
  - actor_lr=1e-6, kl_coef=0.05（adaptive KL）
  - LPPO: ema_beta=0.8, w_min=0.25, w_max=2.0, sigmoid_k=10, tau=0.08

```bash
bash scripts/run_math_grpo.sh<img width="792" height="826" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=6737174512641630031 skey=@crypt_7a1c7503_5e17c462df099ecf983c542a360b9942 mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/d02687cc-b039-457f-bca2-fd6f9bd27554" />

```
**训练结果如下：**
**round1：**
<img width="400" height="400" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=9042519120374792500 skey=@crypt_7a1c7503_5e17c462df099ecf983c542a360b9942 mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/7bd456cc-ba7e-4afd-91ca-78398ebbfc42" />

**round2**：
<img width="400" height="400" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=6737174512641630031 skey=@crypt_7a1c7503_5e17c462df099ecf983c542a360b9942 mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/bc9a9a05-99ba-4294-ae12-6648514eefdf" />


**round3：**
<img width="400" height="400" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=9009613133735928000 skey=@crypt_7a1c7503_5e17c462df099ecf983c542a360b9942 mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/505932c6-de30-4c8c-b6a8-656868842a9f" />




**综合：**
<img width="400" height="400" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=348395094263055489 skey=@crypt_7a1c7503_5e17c462df099ecf983c542a360b9942 mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/01e8e412-cd3e-4fc6-8e5b-d2f27acc9122" />



---

### 7评测（`evaluation/`）

| 文件 | 功能 |
|------|------|
| `eval_math.py` | 评测 GSM8K 和 MATH（支持本地模型 & vLLM API） |
| `eval_math_fewshot.py` | Few-shot 评测 |
| `eval_gsm8k_fewshot.py` | GSM8K Few-shot 专用 |
| `recompute_accuracy.py` | 重新计算已有预测文件的准确率 |

```bash
python3 evaluation/eval_math.py --model /path/to/model --dataset all
```

---

## 环境配置

### 依赖

```bash
pip install -r requirements.txt
```

```
numpy, pandas, pyarrow, tqdm
transformers, vllm, peft, torch, latex2sympy2
```

### 训练框架

| 阶段 | 框架 |
|------|------|
| CPT / SFT / DPO | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| LPPO (GRPO) | [verl](https://github.com/volcengine/verl) |

### 推荐硬件

| 阶段 | 配置 |
|------|------|
| CPT 全参数训练 | 4× A100  + DeepSpeed ZeRO-2 |
| SFT / DPO LoRA | 4× A100 |
| Student 推理 (7B) | 4× A100  (TP=4) |
| Teacher 推理 (32B) | 4× A100  (TP=4) |
| LPPO (GRPO) | 8× A100  |

---

## 注意事项

1. **路径修改**：所有脚本包含原工作环境的绝对路径，克隆后需全局替换。
2. **vLLM 兼容性**：包含 `transformers 5.x` 与 `vllm 0.11.0` 的兼容性 monkey-patch。
3. **数据格式**：题库 JSONL 每行需包含 `question`（题目）和 `ground_truth`（标准答案，纯数值/分数）。
4. **LPPO 模块**：`lppo/` 目录需要从 verl 仓库配合使用，通过 `PYTHONPATH` 引入。



## 引用的模型

- **Base**：[Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) / [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
- **Teacher**： [Qwen2.5-Math-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct)
- **Classifier**：[HuggingFaceTB/openwebmath-classifier](https://huggingface.co/HuggingFaceTB/openwebmath-classifier)
