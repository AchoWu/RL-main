#!/usr/bin/env bash
# ============================================================================
# OPD + Top-K Agreement Policy Gradient (teacher-guided PG sharpening)
# ============================================================================
# Motivation
#   标准 OPD reverse-KL 是 mode-seeking 的全 vocab 拉齐，方差高且对
#   长尾 token 敏感。这里试一个更 RL 味道、更 sparse 的替代信号：
#   把学生 rollout 采到的每个 token 按"学生/教师是否都把它排进 top-k"
#   分成三类，直接做 PPO clipped surrogate 上的策略梯度锐化。
#
# Loss (per sampled token v_t = input_ids[t+1])
#   S_top = 学生当前 forward 的 top-k     (student_k=32)
#   T_top = 教师 top-k                    (teacher_k=32, 来自 topk_logits_k)
#
#     A_t = +1  if v_t ∈ S_top ∩ T_top    (师生都看好, push up)
#     A_t =  0  if v_t ∈ S_top \ T_top    (学生看好、教师不认, 不动)
#     A_t = -1  if v_t ∉ S_top             (罕见, 学生自己都不排进 top-k)
#
#     w_t     = π_θ(v_t) / π_ref(v_t)                π_ref = 初始 ckpt (冻结)
#     L_t     = − min(w_t·A_t, clip(w_t, 0.9, 1.1)·A_t)
#     L       = masked_mean(L_t)
#
# IS ratio + clip 的作用：约束单步更新相对初始 ckpt 的漂移。因为 π_ref 是
# 冻结的初始 ckpt，训练越久 ratio 越会离 1、被 clip 死掉的比例越高，
# 相当于一个软 trust region。诊断指标 topk_pg_is_clipped_*_sum 会暴露
# 这个过程。
# ============================================================================
set -euo pipefail
trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40092/howu/PaddleOCR/test_gpu.py
' EXIT INT TERM

# ====== conda 环境（用 opd，env-3.12 已被清空） ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== 代理（HF / wandb 都需要，容器内默认没有代理变量） ======
: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# ====== HF 缓存（数据集已预下载到这里；模型走 /dev/shm 本地路径） ======
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# ====== 忽略师生词表一致性检查 ======
export NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true

# ====== 保存 checkpoint 超时 = 30 分钟 ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB 在线上传 ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== 让所有 Ray worker 使用当前 conda 环境的 Python，彻底绕过 uv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# 清理旧的 venv 构建残留
rm -rf /group/40092/howu/RL-main/venvs

# 停止残留的 Ray 进程
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== 模型路径（内存盘） ======
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# 实验名
RUN_NAME="opd-skywork7b-to-r1qwen1.5b-topkagreement-pg-s32t32-clip0911"

mkdir -p /group/40092/howu/RL-main/logs

export HTTP_PROXY=http://star-proxy.oa.com:3128
export HTTPS_PROXY=http://star-proxy.oa.com:3128
export WANDB_INIT_TIMEOUT=300

cd /group/40092/howu/RL-main
python examples/run_distillation_math.py \
      --config examples/configs/distillation_math.yaml \
      policy.model_name=/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/ \
      teacher.model_name=/dev/shm/llms/Skywork-OR1-Math-7B/ \
      cluster.gpus_per_node=8 \
      policy.train_micro_batch_size=1 \
      policy.train_global_batch_size=128 \
      policy.optimizer.kwargs.lr=1.0e-6 \
      policy.max_total_sequence_length=32768 \
      teacher.max_total_sequence_length=32768 \
      teacher.logprob_batch_size=1 \
      data.dataset_name=DAPOMath17KProcessed \
      distillation.num_generations_per_prompt=1 \
      distillation.max_num_epochs=1 \
      distillation.max_num_steps=106 \
      distillation.val_period=10 \
      distillation.val_at_start=true \
      distillation.max_val_samples=500 \
      distillation.topk_logits_k=32 \
      data.validation_source=train_holdout \
      data.validation_num_samples=500 \
      data.validation_seed=42 \
      checkpointing.save_period=10 \
      checkpointing.keep_top_k=10 \
      checkpointing.save_consolidated=false \
      checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
      loss_fn.type=topk_agreement_pg \
      loss_fn.topk_agreement_pg.student_k=32 \
      loss_fn.topk_agreement_pg.ratio_clip_min=0.9 \
      loss_fn.topk_agreement_pg.ratio_clip_max=1.1 \
      loss_fn.topk_agreement_pg.reduction=token_mean \
      +data.config_name=en \
      logger.wandb_enabled=true \
      logger.wandb.name="${RUN_NAME}"
