#!/usr/bin/env bash
# Fixed-length teacher-prefix OPD:
#   Skywork-OR1-Math-7B generates a context-only prefix, then
#   DeepSeek-R1-Distill-Qwen-1.5B continues and receives reverse-KL loss only
#   on student-generated targets. Validation always starts the student from 0.
set -euo pipefail

PREFIX_LEN="${1:-${TEACHER_PREFIX_LENGTH:-0}}"
case "$PREFIX_LEN" in
  0|256|512|1024) ;;
  *)
    echo "Usage: $0 {0|256|512|1024}"
    exit 2
    ;;
esac

# ====== conda environment ======
source /data/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate opd
echo "▶ Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

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

# The two Qwen-family tokenizers have the same token mapping, but padded model
# vocab sizes differ (teacher 152064, student 151936).
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

rm -rf /group/40092/howu/RL-main/venvs
ray stop --force 2>/dev/null || true

sed -i 's/PY_EXECUTABLES.AUTOMODEL/PY_EXECUTABLES.SYSTEM/; s/PY_EXECUTABLES.FSDP/PY_EXECUTABLES.SYSTEM/' /group/40092/howu/RL-main/nemo_rl/distributed/ray_actor_environment_registry.py
export PYTHONPATH=/group/40092/howu/RL-main:${PYTHONPATH:-}

# ====== local models ======
TEACHER_MODEL="/dev/shm/llms/Skywork-OR1-Math-7B/"
POLICY_MODEL="/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
for d in "$TEACHER_MODEL" "$POLICY_MODEL"; do
  if [[ ! -f "${d}config.json" ]]; then
    echo "Model missing: ${d}config.json"
    exit 1
  fi
done
echo "▶ Using policy=$POLICY_MODEL teacher=$TEACHER_MODEL prefix=$PREFIX_LEN"

RUN_NAME="opd-skywork7b-to-r1qwen1.5b-teacher-prefix-p${PREFIX_LEN}-reverse-100step"
mkdir -p /group/40092/howu/RL-main/logs

cd /group/40092/howu/RL-main
python examples/run_distillation_math.py \
    --config examples/configs/distillation_math.yaml \
    policy.model_name="$POLICY_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    cluster.gpus_per_node=8 \
    policy.train_micro_batch_size=1 \
    policy.train_global_batch_size=128 \
    policy.optimizer.kwargs.lr=1.0e-6 \
    teacher.logprob_batch_size=1 \
    distillation.teacher_prefix_length="$PREFIX_LEN" \
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

python test_gpu.py
