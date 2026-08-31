#!/usr/bin/env bash
# ============================================================================
# OPD + EMA-of-student anchor
# ============================================================================
# Idea
#   Reverse-KL to the teacher as usual, plus an auxiliary reverse-KL to a
#   trailing EMA of the student's own weights:
#       L = KL(p_S || p_T)  +  lambda * KL(p_S || p_EMA)
#       theta_ema <- mu * theta_ema + (1 - mu) * theta_student  (per step)
#
# Why (in one sentence): reference-KL against the frozen initial student is a
#   brake (pulls student back), EMA-anchor is a shock absorber (only resists
#   high-frequency change, tracks slow progress). It targets the rollout
#   instability we see in 32k long-CoT training rather than "which tokens to
#   learn", which every prior gate has been unable to move.
#
# Physical reuse
#   The existing `reference_model_state_dict` slot (CPU-resident, allocated
#   when init_reference_model=True) is reused as the EMA buffer. Distillation
#   flips it from "frozen snapshot" to "EMA-updated snapshot" via a new
#   worker method `update_reference_ema(mu)`, and pulls top-k logits under
#   those weights via `get_reference_topk_logits`. Mutually exclusive with
#   reference_policy_kl_penalty.
#
# Cost per step
#   +1 forward of the 1.5B student (top-k logits under EMA weights) via the
#   existing use_reference_model context manager. In practice negligible next
#   to the 7B teacher forward. No extra GPU memory beyond the CPU buffer that
#   was already allocated for reference-KL.
# ============================================================================
set -euo pipefail
trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40092/howu/PaddleOCR/test_gpu.py
' EXIT INT TERM

# ====== conda 环境 ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== 代理 ======
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

# ====== NCCL / vLLM 超时 ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== WandB ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation

# ====== Ray / venv ======
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
rm -rf /group/40092/howu/RL-main/venvs
ray stop --force 2>/dev/null || true
sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' \
    /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py

export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== 模型路径 ======
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "❌ 模型缺失: ${d}config.json 不存在"; exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL  teacher=$TEACHER_MODEL"

# ====== EMA anchor 超参 ======
# mu       —— EMA decay. 0.999 ≈ 1000-step memory (covers the whole 106-step run
#              as "trailing average of student"). Drop to 0.995 for a
#              ~200-step window if the anchor feels too sticky.
# kl_weight —— lambda on KL(p_S || p_EMA). Watch wandb: ema_anchor_kl /
#              distillation_loss should sit around 0.1–0.3 for the anchor to
#              have a meaningful but non-dominant effect.
EMA_MU=0.999
EMA_KL_WEIGHT=0.05

RUN_NAME="opd-skywork7b-to-r1qwen1.5b-ema-anchor-mu${EMA_MU}-w${EMA_KL_WEIGHT}-reverse-260831"

mkdir -p /group/40092/howu/RL-main/logs
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
      data.validation_source=train_holdout \
      data.validation_num_samples=500 \
      data.validation_seed=42 \
      checkpointing.save_period=10 \
      checkpointing.keep_top_k=10 \
      checkpointing.save_consolidated=false \
      checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
      loss_fn.kl_type=reverse \
      loss_fn.zero_outside_topk=false \
      +loss_fn.ema_anchor.enabled=true \
      +loss_fn.ema_anchor.mu=${EMA_MU} \
      +loss_fn.ema_anchor.kl_weight=${EMA_KL_WEIGHT} \
      +data.config_name=en \
      logger.wandb_enabled=true \
      logger.wandb.name="${RUN_NAME}"
