#!/usr/bin/env bash
# ============================================================================
# OPD + Top-K Agreement Weighted-CE (vocabulary-level, no IS, no clip)
# ============================================================================
# Motivation
#   topk_agreement_pg 是 token-level REINFORCE：每个位置只在采样到的 v_t
#   一个 token 上求梯度，加上 IS+clip 之后 push-up 在 ~step 10 就饱和，
#   信号密度不够拉不动 baseline (~0.6+)，peak 只到 0.524。
#
#   这个 CE 变式换成词表级：每位置对学生 top-k_S 的所有 k_S 个 token
#   都算 log_p 并加权，密度提高 ~k_S 倍。同时因为没有采样步骤，
#   IS ratio + PPO clip 自动消失。
#
# Loss (per context position t)
#   S_top = 学生 top-k_S  (student_k=32)
#   T_top = 教师 top-k_T  (topk_logits_k=32)
#
#     L_pos_t = − mean_{v ∈ S_top ∩ T_top}  log π_θ(v | x_<t)   # push up
#     L_neg_t = + mean_{v ∈ S_top \ T_top}  log π_θ(v | x_<t)   # push down
#     L_t     = L_pos_t + push_down_weight · L_neg_t
#     L       = masked_mean(L_t)
#
# 语义：让学生 top-k 里"教师也认可"的那批 token 概率被拉高，
# "学生自己看好但教师没排 top-k"的那批被压低。log π_θ(v) 用全词表
# 归一化(logits − logsumexp)，不是 top-k 内部重归一化。
# π_ref 不再参与 loss 计算——纯监督形式，靠 max_grad_norm 保稳定。
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

# ====== HF 缓存 ======
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
RUN_NAME="opd-skywork7b-to-r1qwen1.5b-topkagreement-ce-s32t32-pd1.0"

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
      loss_fn.type=topk_agreement_ce \
      loss_fn.topk_agreement_ce.student_k=32 \
      loss_fn.topk_agreement_ce.push_down_weight=1.0 \
      loss_fn.topk_agreement_ce.reduction=token_mean \
      +data.config_name=en \
      logger.wandb_enabled=true \
      logger.wandb.name="${RUN_NAME}"
