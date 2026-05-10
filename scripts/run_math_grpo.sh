#!/usr/bin/env bash
# 数学 RLVR (GRPO) 训练 Pipeline
#
# 基于 verl 框架，使用 Qwen2.5-Math-7B-WarmStart 作为起点
# 8×A100-40GB，vLLM rollout，GRPO 算法
#
# 与 TTS 版 run_v3_hf.sh 的区别：
#   - 模型：Qwen2.5-Math-7B-WarmStart（而非 CosyVoice3）
#   - Reward：verl 内置 math_dapo（答案正确性），无需外部 reward server
#   - Response length：2048（数学推理需要更长）
#   - 无 duration reward（W_DUR=0.0）
#   - 无需 reward server，也不需要自定义 reward function

set -eou pipefail

stage=2
stop_stage=2

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# ===== 路径配置 =====
BASE_DIR=/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline
VERL_DIR=/cfs/cfs-esygraib/belvathliu/cv3/verl

# 起点模型：从 WarmStart 继续做 GRPO
model_path=${BASE_DIR}/Qwen2.5-Math-7B-WarmStart
if [ ! -d "$model_path" ]; then
  log "WarmStart 模型不存在: $model_path"
  exit 1
fi
log "使用模型: $model_path"

data_dir=${BASE_DIR}/data/parquet_math_rlvr_v4_balanced
exp_name=qwen25_math_grpo_lppo_v4_20260506

# ===== Stage 1: 准备数据 =====
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: 准备数学 RLVR 训练数据"
  python3 ${BASE_DIR}/prepare_math_rlvr_data.py \
    --extracted_qa ${BASE_DIR}/sft_data/dpo_questions_120k_expanded_v4.jsonl \
    --student_responses ${BASE_DIR}/sft_data/student_responses_120k_expanded_v4.jsonl \
    --teacher_responses ${BASE_DIR}/sft_data/teacher_responses_48k_final_v4.jsonl \
    --output_dir $data_dir \
    --total 0 \
    --balance_categories \
    --num_test 1000 \
    --seed 42

  # === LPPO: 初始化 LP State（从 student_responses 计算初始 P0）===
  # 原理：用历史采样结果作为先验，避免训练初期的"虚假学习进度"
  log "stage 1.5: 初始化 LPPO Learning Progress 状态"
  python3 -m lppo.lp_init \
    --student_responses ${BASE_DIR}/sft_data/student_responses_120k_expanded_v4.jsonl \
    --output ${BASE_DIR}/checkpoints/$exp_name/lp_state.json \
    --beta 0.8 \
    --w_min 0.25 \
    --w_max 2.0
fi

# ===== Stage 2: GRPO 训练 =====
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: GRPO 训练 (vLLM rollout, 8 GPUs)"

  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export MKL_SERVICE_FORCE_INTEL=TRUE
  export RAY_DEDUP_LOGS=0
  # 不使用 duration reward（TTS 专用）
  export W_DUR=0.0
  rollout_n=8
  export ROLLOUT_N=$rollout_n
  # OpenWebMath CoT 质量奖励
  export LAMBDA_FM=0.05
  export FM_GATE_THRESHOLD=3.0
  export FM_MODEL_PATH=/cfs/cfs-esygraib/belvathliu/ttmp/floders/pipline/openwebmath-classifier
  # Wrong answers should not dominate GRPO advantages. Keep boxed-but-wrong at
  # 0, and only lightly penalize unformatted wrong answers.
  export WRONG_BOXED_PENALTY=0.0
  export WRONG_UNFORMATTED_PENALTY=-0.1
  export NEAR_CORRECT_REL_TOL=0.01
  export NEAR_CORRECT_ABS_TOL=1e-4
  export MAX_REPETITION_PENALTY=0.05
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  # 添加 verl 和 reward function 到 PYTHONPATH
  export PYTHONPATH="${VERL_DIR}:${BASE_DIR}:${PYTHONPATH:-}"

  n_gpus_per_node=8
  micro_batch_size=1    # 先维持 1，降低 actor/ref 侧显存压力
  train_batch_size=8    # 配合 rollout.n=8，有效 real_train_batch_size=64
  ppo_mini_batch_size=8 # 配合 rollout.n=8，有效 actor mini_batch_size=64，可被 DP=8 整除
  dataloader_num_workers=0  # 避免 StatefulDataLoader 子进程在验证/退出阶段被系统 kill
  val_batch_size=64         # verl 默认会把整个验证集作为一个 batch，这里显式分块

  LOG_FILE=${BASE_DIR}/${exp_name}.log
  log "日志输出到: $LOG_FILE"
  log "8卡配置: train_batch_size=$train_batch_size, rollout_n=$rollout_n, ppo_mini_batch_size=$ppo_mini_batch_size, micro_batch_size=$micro_batch_size, dataloader_num_workers=$dataloader_num_workers, val_batch_size=$val_batch_size"

  # 8 卡时，actor 的有效 mini_batch_size 需要能被 DP=8 整除
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=$data_dir/train.parquet \
      data.val_files=$data_dir/test.parquet \
      data.train_batch_size=$train_batch_size \
      ++data.dataloader_num_workers=$dataloader_num_workers \
      ++data.val_batch_size=$val_batch_size \
      ++data.validation_shuffle=False \
      data.max_prompt_length=512 \
      data.max_response_length=1024 \
      data.truncation='left' \
      actor_rollout_ref.rollout.response_length=1024 \
      actor_rollout_ref.model.use_remove_padding=False \
      actor_rollout_ref.model.path=$model_path \
      ++actor_rollout_ref.rollout.dtype=bfloat16 \
      ++actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
      ++actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.05 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      algorithm.kl_ctrl.type=adaptive \
      algorithm.kl_ctrl.kl_coef=0.05 \
      algorithm.kl_ctrl.target_kl=0.1 \
      algorithm.kl_ctrl.horizon=10000 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
      actor_rollout_ref.rollout.do_sample=true \
      actor_rollout_ref.rollout.temperature=0.7 \
      actor_rollout_ref.rollout.top_p=0.9 \
      actor_rollout_ref.rollout.n=$rollout_n \
      actor_rollout_ref.rollout.val_kwargs.do_sample=true \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
      reward_model.reward_manager=naive \
      custom_reward_function.path=reward_math_rlvr.py \
      custom_reward_function.name=compute_score \
      trainer.project_name='qwen25_math_grpo' \
      trainer.experiment_name=$exp_name \
      trainer.logger=['console','tensorboard'] \
      +trainer.log_dir=${BASE_DIR}/tensorboard_logs/$exp_name \
      trainer.n_gpus_per_node=$n_gpus_per_node \
      trainer.nnodes=1 \
      trainer.save_freq=300 \
      trainer.max_actor_ckpt_to_keep=1 \
      trainer.test_freq=200 \
      trainer.resume_mode='auto' \
      trainer.default_local_dir=${BASE_DIR}/checkpoints/$exp_name \
      trainer.total_epochs=1 \
      trainer.val_before_train=True \
      ++lppo.enable=True \
      ++lppo.state_path=${BASE_DIR}/checkpoints/$exp_name/lp_state.json \
      ++lppo.module_path=${BASE_DIR} \
      ++lppo.ema_beta=0.8 \
      ++lppo.w_min=0.25 \
      ++lppo.w_max=2.0 \
      ++lppo.sigmoid_k=10 \
      ++lppo.tau=0.08 \
      2>&1 | tee $LOG_FILE
fi

# ===== Stage 3: 评估 =====
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "stage 3: 评估 GRPO 训练后的模型"
  # 需要先合并 checkpoint，然后用 eval 脚本评测
  # 具体步骤参考 run_v3_hf.sh 的 Stage 3-5
  echo "TODO: 合并 checkpoint 并评测 GSM8K + MATH"
fi
