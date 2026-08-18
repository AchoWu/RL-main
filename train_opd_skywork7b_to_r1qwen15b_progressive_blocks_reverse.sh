#!/usr/bin/env bash
# Four-stage progressive teacher-block OPD:
#   steps  0-24: student generates/trains on tokens   0-256
#   steps 25-49: teacher supplies 0-256, student trains on 256-512
#   steps 50-74: teacher supplies 0-512, student trains on 512-768
#   steps 75-99: teacher supplies 0-768, student trains on 768-1024
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/group/40094/jingweidong/RL-main}"

# ====== conda environment ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ====== proxy ======
: "${ENV_VENUS_PROXY:=http://star-proxy.oa.com:3128}"
export NO_PROXY=localhost,127.0.0.1,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# ====== Hugging Face cache ======
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

export NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true

# ====== distributed/checkpoint timeouts ======
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800

# ====== logging/runtime ======
export WANDB_MODE=online
export WANDB_PROJECT=nemo-distillation
export WANDB_INIT_TIMEOUT=300
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

rm -rf "${REPO_ROOT}/venvs"
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' \
    "${REPO_ROOT}/nemo_rl/distributed/ray_actor_environment_registry.py"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ====== local models ======
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "Model missing: ${d}config.json"
    exit 1
  fi
done

RUN_NAME="opd-skywork7b-to-r1qwen1.5b-progressive-block256x4-reverse-100step"
mkdir -p "${REPO_ROOT}/logs"
echo "Using policy=$POLICY_MODEL teacher=$TEACHER_MODEL"
echo "Curriculum: 4 stages x 25 steps, 256 student tokens per stage"

cd "$REPO_ROOT"
python examples/run_distillation_math.py \
    --config examples/configs/distillation_math.yaml \
    policy.model_name="$POLICY_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    policy.train_global_batch_size=128 \
    policy.optimizer.kwargs.lr=1.0e-6 \
    policy.generation.vllm_cfg.async_engine=false \
    teacher.logprob_batch_size=1 \
    distillation.teacher_prefix_length=0 \
    distillation.progressive_teacher_blocks.enabled=true \
    distillation.progressive_teacher_blocks.block_size=256 \
    distillation.progressive_teacher_blocks.steps_per_stage=25 \
    distillation.max_num_epochs=1 \
    distillation.max_num_steps=100 \
    distillation.val_period=10 \
    distillation.val_at_start=true \
    distillation.max_val_samples=1000 \
    data.validation_source=train_holdout \
    data.validation_num_samples=1000 \
    data.validation_seed=42 \
    checkpointing.save_period=10 \
    checkpointing.metric_name=null \
    checkpointing.keep_top_k=2 \
    checkpointing.save_consolidated=false \
    checkpointing.checkpoint_dir="checkpoints/distillation-${RUN_NAME}" \
    logger.wandb_enabled=true \
    loss_fn.kl_type=reverse \
    logger.wandb.name="${RUN_NAME}"
